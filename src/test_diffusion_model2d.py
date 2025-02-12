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
    w = torch.tensor(rng.normal(size=(batchsize, m, m)))
    alpha_hat = torch.tensor(rng.normal(size=(batchsize, m + 1, m + 1)))
    u.backward(gradient=w)
    print(alpha_hat.shape, alpha.grad.shape)
    print(torch.sum(alpha_hat * alpha.grad, dim=[-2, -1]))
    epsilon = 1.0e-6
    alpha = alpha.detach()

    manual_gradient = (model(alpha + epsilon * alpha_hat) - model(alpha)) / epsilon
    print(torch.sum(w * manual_gradient, dim=[-2, -1]))
