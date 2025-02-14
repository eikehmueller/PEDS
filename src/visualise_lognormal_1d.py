import itertools
import numpy as np
from matplotlib import pyplot as plt
from peds.distributions import matern, LogNormalDistribution1d

n = 256
Lambda = 0.1  # correlation length
a_power = 2
nu = a_power - 1 / 2

distribution = LogNormalDistribution1d(n, Lambda, a_power)

# Compute covariance estimator
n_samples = 10000
cov = np.zeros(n + 1)
var = 0

for alpha in itertools.islice(iter(distribution), n_samples):
    for j in range(n + 1):
        cov[j] += alpha[j] * alpha[3 * n // 4] / n_samples
    var += sum(alpha[:] ** 2) / ((n + 1) * n_samples)
print(f"var = {var}")
plt.clf()
X = np.arange(0, 1 + 0.5 / n, 1 / n)
for alpha in itertools.islice(iter(distribution), 8):
    plt.plot(X, alpha, linewidth=2)
ax = plt.gca()
ax.set_xlabel("x")
ax.set_ylabel(r"$\alpha(x)$")
plt.savefig("samples.pdf", bbox_inches="tight")


plt.clf()
plt.plot(
    X,
    cov,
    label=r"estimator of $\sigma^{-2}\mathbb{E}[\alpha(x)\alpha(\frac{3}{4})]$",
)
plt.plot(
    X,
    matern((X - 0.75) / Lambda, nu),
    label=r"matern covariance $\sigma^{-2}C_{\nu,\kappa}(\kappa|x-\frac{3}{4}|)$",
)
plt.legend(loc="upper left")
ax = plt.gca()
ax.set_xlabel("x")
plt.savefig("covariance.pdf", bbox_inches="tight")
