import numpy as np
import torch
from peds.diffusion_model_2d import Solver2d, DiffusionModel2d


def test_solver2d():
    rng = np.random.default_rng(seed=215725)
    m = 8
    f_rhs = rng.normal(size=(m, m))
    alpha = rng.normal(size=(m + 1, m + 1))
    solver = Solver2d(m)
    u = solver.solve(alpha, f_rhs)


def test_diffusion_model_2d():
    rng = np.random.default_rng(seed=215725)
    batchsize = 4
    m = 8
    f_rhs = rng.normal(size=(m, m))
    alpha = torch.tensor(rng.normal(size=(batchsize, m + 1, m + 1)), requires_grad=True)
    model = DiffusionModel2d(f_rhs)
    u = model(alpha)
    external_grad = torch.ones_like(u)
    u.backward(gradient=external_grad)
    print(alpha.grad)
