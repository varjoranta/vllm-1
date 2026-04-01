# SPDX-License-Identifier: Apache-2.0
"""Lloyd-Max optimal centroids for N(0, 1/d) Gaussian distribution.

Precomputed codebooks for TurboQuant. These are data-oblivious: the same
centroids work for any model because after WHT rotation, coordinates
are approximately Gaussian with variance 1/d.
"""

import math

from scipy import stats


def lloyd_max_centroids(n_centroids: int, sigma: float,
                        n_iter: int = 100) -> list[float]:
    """Compute optimal scalar quantization centroids via Lloyd's algorithm.

    Args:
        n_centroids: number of centroids (2^bits)
        sigma: standard deviation of the Gaussian distribution
        n_iter: number of Lloyd iterations

    Returns:
        Sorted list of centroid values
    """
    if n_centroids == 2:
        c = math.sqrt(2.0 / math.pi) * sigma
        return [-c, c]

    if n_centroids == 4:
        centroids = [-1.51 * sigma, -0.453 * sigma, 0.453 * sigma, 1.51 * sigma]
        return centroids

    # Initialize boundaries from quantiles
    boundaries = [float(stats.norm.ppf(i / n_centroids, scale=sigma))
                  for i in range(1, n_centroids)]
    centroids = [0.0] * n_centroids

    def cond_exp(a: float, b: float) -> float:
        a_s = a / sigma if math.isfinite(a) else a
        b_s = b / sigma if math.isfinite(b) else b
        if not math.isfinite(a_s):
            prob = stats.norm.cdf(b_s)
        elif not math.isfinite(b_s):
            prob = stats.norm.sf(a_s)
        else:
            prob = stats.norm.cdf(b_s) - stats.norm.cdf(a_s)
        if prob < 1e-15:
            return ((a if math.isfinite(a) else 0) +
                    (b if math.isfinite(b) else 0)) / 2
        return sigma * (stats.norm.pdf(a_s) - stats.norm.pdf(b_s)) / prob

    for _ in range(n_iter):
        centroids[0] = cond_exp(-math.inf, boundaries[0])
        for i in range(1, n_centroids - 1):
            centroids[i] = cond_exp(boundaries[i - 1], boundaries[i])
        centroids[-1] = cond_exp(boundaries[-1], math.inf)
        boundaries = [(centroids[i] + centroids[i + 1]) / 2
                      for i in range(n_centroids - 1)]

    centroids.sort()
    return centroids


def optimal_centroids(bits: int, dim: int) -> list[float]:
    """Get optimal centroids for N(0, 1/d) at given bit width."""
    sigma = 1.0 / math.sqrt(dim)
    return lloyd_max_centroids(1 << bits, sigma)
