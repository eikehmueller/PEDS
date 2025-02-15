import itertools
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from peds.distributions import matern, LogNormalDistribution2d
from firedrake import *


def save_vtk(alpha_samples, filename):
    """Save a list of samples to vtk file

    :arg alpha_samples: list of fields
    :arg filename: name of file to write to
    """
    alpha = alpha_samples[0]
    nx = alpha.shape[0] - 1
    ny = alpha.shape[1] - 1
    npoints = (nx + 1) * (ny + 1)
    hx = 1.0 / nx
    hy = 1.0 / ny
    with open(filename, "w", encoding="utf8") as f:
        print("# vtk DataFile Version 2.0", file=f)
        print("Sample state", file=f)
        print("ASCII", file=f)
        print("DATASET STRUCTURED_POINTS", file=f)
        print(f"DIMENSIONS {nx + 1} {ny + 1} 1 ", file=f)
        print("ORIGIN -0.5 -0.5 0.0", file=f)
        print(f"SPACING {hx} {hy} 0", file=f)
        print(file=f)
        print(f"POINT_DATA {npoints}", file=f)
        for ell, alpha in enumerate(alpha_samples):
            print(f"SCALARS sample_{ell:03d} double 1", file=f)
            print("LOOKUP_TABLE default", file=f)
            for j in range(nx + 1):
                for k in range(nx + 1):
                    print(f"{alpha[j,k]:12.8e}", file=f)


n = 64
Lambda = 0.1  # correlation length
nu = 1

distribution = LogNormalDistribution2d(n, Lambda)

# Compute covariance estimator
n_samples = 10000

j0, k0 = n // 2, n // 2

d0 = 1e-3
dmax = d0 + max(
    np.sqrt(j0**2 + k0**2) / n,
    np.sqrt((j0 - n) ** 2 + k0**2) / n,
    np.sqrt(j0**2 + (k0 - n) ** 2) / n,
    np.sqrt((j0 - n) ** 2 + (k0 - n) ** 2) / n,
)

nbins = 40
cov = np.zeros(nbins + 1)
bin_volume = np.zeros(nbins + 1)
var = 0

for j in range(n + 1):
    for k in range(n + 1):
        d = np.sqrt((j - j0) ** 2 + (k - k0) ** 2) / n
        bin_idx = int(np.floor(nbins * d / dmax))
        bin_volume[bin_idx] += 1

alpha_samples = []
for ell, alpha in enumerate(
    tqdm(itertools.islice(iter(distribution), n_samples), total=n_samples)
):
    if ell < 16:
        alpha_samples.append(alpha)
    for j in range(n + 1):
        for k in range(n + 1):
            d = np.sqrt((j - j0) ** 2 + (k - k0) ** 2) / n
            bin_idx = int(np.floor(nbins * d / dmax))
            cov[bin_idx + 1] += (
                alpha[j0, k0] * alpha[j, k] / (n_samples * bin_volume[bin_idx])
            )
    cov[0] += alpha[j0, k0] ** 2 / n_samples
save_vtk(alpha_samples, "samples.vtk")
print(f"variance = {cov[0]}")
plt.clf()
X = (np.arange(nbins + 1) - 0.5) / nbins * dmax
X[0] = 0
plt.plot(
    X,
    cov,
    marker="o",
    markersize=4,
    label=r"estimator of $\sigma^{-2}\mathbb{E}[\alpha(x,y)\alpha(x_0,y_0)]$",
)
Y = matern((X / Lambda), 1)
Y[0] = 1
plt.plot(
    X,
    Y,
    label=r"matern covariance $\sigma^{-2}C_{1,\kappa}(\kappa||x-x_0||)$",
)
plt.legend(loc="upper right")
ax = plt.gca()
ax.set_xlabel("$||x-x_0||$")
plt.savefig("covariance_2d.pdf", bbox_inches="tight")
