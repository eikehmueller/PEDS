import torch

import numpy as np
from matplotlib import pyplot as plt

from peds.diffusion_model import DiffusionModel1d
from peds.distributions import LogNormalDistribution1d

torch.set_default_dtype(torch.float64)

rng = torch.Generator()
rng.manual_seed(25157)

batch_size = 8
n = 128
f_rhs = torch.ones(size=(n,))
distribution = LogNormalDistribution1d(n, Lambda=0.1, a_power=2)

X_alpha = np.arange(0, 1 + 0.5 / n, 1 / n)
X_u = np.arange(0, 1, 1 / n) + 0.5 / n

fig, axs = plt.subplots(nrows=1, ncols=2)
for j in range(batch_size):
    alpha = next(iter(distribution))
    model = DiffusionModel1d(f_rhs)
    u = model(torch.Tensor(alpha))
    axs[0].plot(X_alpha, np.exp(alpha))
    axs[1].plot(X_u, u)
for ax in axs:
    ax.set_xlim(0, 1)
    ax.set_xlabel("$x$")
axs[0].set_ylabel("diffusion coefficient $K(x)$")
axs[1].set_ylabel("solution $u(x)$")

plt.savefig("solution_sample.pdf", bbox_inches="tight")
