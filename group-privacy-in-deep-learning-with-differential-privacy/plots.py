import numpy as np
from scipy import optimize
from scipy.stats import hypergeom, norm

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent

from matplotlib import pyplot as plt

from probabilitybuckets_light import ProbabilityBuckets as PB


# required hyper parameters, here set for a typical MNIST example
N = 60000  # populations size
k = 5      # samples to protect
batch_size = 256    # typical batchsize
target_delta = 1e-5

# the clipping bound C is set to 1 for the full evaluation
sigma = 1

# Probablity Buckets parameter, see GitHub repo for explanation
number_of_buckets = 100000


def compute_epsilon(steps, k, N, noise_multiplier, B, target_delta=1e-5):
    """Computes epsilon value for given hyperparameters for more than one protected sample. Code is partially copied from tensorflow privacy project."""
    if noise_multiplier == 0.0:
        return float('inf')
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    q = 1 - hypergeom.cdf(0, N, k, B)  # This is a vast overapproximation!
    rdp = compute_rdp(
        q=q,
        # our sensitivity is not 1 but j*1. We scale the x-axis accoridngly by dividing the noise_multiplier by k
        noise_multiplier=noise_multiplier / k,
        steps=steps,
        orders=orders)
    # Delta is set to 1e-5 because MNIST has 60000 training points.
    return get_privacy_spent(orders, rdp, target_delta=target_delta)[0]


# # The original implementation from thensorlfow privacy. Yields the same result as 'compute_epsilon' for k=1
# def compute_epsilon_original(steps, noise_multiplier, batch_size):
#   """Computes epsilon value for given hyperparameters."""
#   if noise_multiplier == 0.0:
#     return float('inf')
#   orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
#   sampling_probability = batch_size / 60000
#   rdp = compute_rdp(q=sampling_probability,
#                     noise_multiplier=noise_multiplier,
#                     steps=steps,
#                     orders=orders)
#   # Delta is set to 1e-5 because MNIST has 60000 training points.
#   return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]


def get_worst_case_distributions(k, batch_size, N, sigma, truncation_multiplier=50, granularity=10000, return_x=False):
    width = sigma * truncation_multiplier
    # the points on the x-axis we generate discrete noise for.
    x = np.linspace(-width, width + k * sigma, int((2 * width + k * sigma) * granularity))

    # the first distribution
    A = norm.pdf(x, loc=0, scale=sigma)

    B = np.zeros_like(A)
    probs = hypergeom.pmf(np.arange(0, k + 1), N, k, batch_size)

    for j in range(k + 1):
        B += probs[j] * norm.pdf(x, loc=j, scale=sigma)

    A /= np.sum(A)
    B /= np.sum(B)  # normalise due to discretisation

    if return_x:
        return A, B, x
    return A, B


def get_PB_eps(k, batch_size, N, sigma, target_delta, number_of_compositions, factor=1 + 1e-6):

    A, B = get_worst_case_distributions(k, batch_size, N, sigma)

    # Initialize privacy buckets.
    kwargs = {'number_of_buckets': number_of_buckets,
              'factor': factor,
              'caching_directory': "./pb-cache",  # caching makes re-evaluations faster. Can be turned off for some cases.
              'free_infty_budget': 10**(-20),  # how much we can put in the infty bucket before first squaring
              'error_correction': True,
              }

    pb = PB(dist1_array=A,  # distribution A
            dist2_array=B,  # distribution B
            **kwargs)

    epses = []
    for comps in number_of_compositions:

        # input can be arbitrary positive integer, but exponents of 2 are numerically the most stable
        pb_composed = pb.compose(comps)

        max_eps = (pb_composed.number_of_buckets // 2 ) * pb_composed.log_factor   # some PB internals, not really relevant

        try:
            root = optimize.bisect(lambda eps: pb_composed.delta_ADP_upper_bound(eps) - target_delta, 0.0001, max_eps)
        except ValueError as e:
            pb_composed.print_state()
            plt.plot(pb_composed.bucket_distribution)
            plt.show()
            raise e
        print("[*]", root)
        epses.append(root)

    return epses


# ##### Plot of distributions

if True:
    da_sigma = 0.1
    da_k = 10
    A, B, x = get_worst_case_distributions(k=da_k, batch_size=batch_size, N=N, sigma=da_sigma, truncation_multiplier=140, granularity=100, return_x=True)

    plt.semilogy(x, A, label="A")
    plt.semilogy(x, B, label=f"B with k={da_k}")
    plt.ylabel("pdf")
    plt.xlabel("x")
    plt.xlim(-1.5, 11)
    plt.ylim(1e-35, 2e-1)
    plt.legend()
    plt.show()


# ##### Plot illustrating composition


number_compositions_MINST = N * 60 / batch_size  # N samples * 60 epochs / batch_size

intermediate_compositions = 2**np.arange(0, np.ceil(np.log2(number_compositions_MINST)) + 1, dtype=np.int64)
print(intermediate_compositions)

da_ks = [1, 2, 3, 5, 10]

if True:
    for k in da_ks:
        e = get_PB_eps(k, batch_size, N, sigma, target_delta, intermediate_compositions)
        print(e)
        plt.semilogx(intermediate_compositions, e, label=f"PB method, k = {k}, sigma = {sigma}")


    for k in da_ks[:-2]:
        epses = []
        for comp in intermediate_compositions:
            print(comp)
            eps = compute_epsilon(steps=comp, k=k, N=N, noise_multiplier=sigma, B=batch_size, target_delta=target_delta)
            # # the implementation from tensorflow-privacy examples yield the same results for k=1
            # eps = compute_epsilon_original(steps=comp, noise_multiplier=sigma, batch_size=batch_size)
            epses.append(eps)
        plt.semilogx(intermediate_compositions, epses, '--', label=f"RDP method, k = {k}, sigma = {sigma}")
        print(epses)

    plt.xlabel("number of compositions")
    plt.ylabel("eps")
    plt.ylim(0, 60)
    plt.legend()

    plt.show()


# ##### Plot illustrating k

if True:
    number_compositions = int(2**np.ceil(np.log2(number_compositions_MINST)))
    da_sigmas = [1, 1.2, 1.5, 2]

    da_ks = np.arange(1, 12)
    for sigma in da_sigmas:
        epses = []
        for k in da_ks:
            print("k: ", k, sep="")
            eps = get_PB_eps(k, batch_size, N, sigma, target_delta, [number_compositions])
            epses.append(eps)
        plt.plot(da_ks, epses, label=f"PB method, sigma = {sigma}")


    for sigma in da_sigmas:
        epses = []
        for k in da_ks:
            eps = compute_epsilon(steps=number_compositions, k=k, N=N, noise_multiplier=sigma, B=batch_size, target_delta=target_delta)
            epses.append(eps)
        plt.plot(da_ks, epses, '--', label=f"RDP method, sigma = {sigma}")

    plt.xlabel("number protected samples k")
    plt.ylabel("eps")
    plt.grid()
    plt.ylim(0, 50)
    plt.legend()
    plt.show()
