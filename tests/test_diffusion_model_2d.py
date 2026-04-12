import numpy as np
import torch
from peds.diffusion_model_2d import Solver2d, DiffusionModel2d


def test_solver_2d():
    """Check that the solver for the 2d diffusion problem works"""
    rng = np.random.default_rng(seed=215725)
    m = 16
    domain_size = 0.7
    solver = Solver2d(m, domain_size)
    u_exact = rng.normal(size=(m, m))
    alpha = rng.normal(size=(m + 1, m + 1))
    f_rhs = solver.apply_operator(alpha, u_exact)
    u = solver.solve(alpha, f_rhs)
    tolerance = 1e-12
    difference = np.linalg.norm(u - u_exact)
    assert difference < tolerance


def test_diffusion_model_2d_gradient():
    """Check that the gradient of the 2d model agrees with finite difference approximation"""
    rng = np.random.default_rng(seed=231575)
    m = 64
    domain_size = 0.7
    f_rhs = rng.normal(size=(m, m))
    alpha = torch.tensor(rng.normal(size=(m + 1, m + 1)), requires_grad=True)
    model = DiffusionModel2d(f_rhs, domain_size)
    u = model(alpha)
    w = torch.tensor(rng.normal(size=(m, m)))
    alpha_hat = torch.tensor(rng.normal(size=(m + 1, m + 1)))
    u.backward(gradient=w)
    delta_torch = torch.sum(alpha_hat * alpha.grad, dim=[-2, -1])
    epsilon = 1.0e-6
    manual_gradient = (model(alpha + epsilon * alpha_hat) - model(alpha)) / epsilon
    delta_manual = torch.sum(w * manual_gradient, dim=[-2, -1])
    difference = np.linalg.norm((delta_torch - delta_manual).detach().numpy())
    assert difference < epsilon


def test_diffusion_model_2d_gradient_batched():
    """Check that the gradient of the 2d model agrees with finite difference approximation

    This test works on batched data
    """
    rng = np.random.default_rng(seed=231575)
    batchsize = 16
    m = 32
    domain_size = 0.7
    f_rhs = rng.normal(size=(m, m))
    alpha = torch.tensor(rng.normal(size=(batchsize, m + 1, m + 1)), requires_grad=True)
    model = DiffusionModel2d(f_rhs, domain_size)
    u = model(alpha)
    w = torch.tensor(rng.normal(size=(batchsize, m, m)))
    alpha_hat = torch.tensor(rng.normal(size=(batchsize, m + 1, m + 1)))
    u.backward(gradient=w)
    delta_torch = torch.sum(alpha_hat * alpha.grad, dim=[-2, -1])
    epsilon = 1.0e-6
    manual_gradient = (model(alpha + epsilon * alpha_hat) - model(alpha)) / epsilon
    delta_manual = torch.sum(w * manual_gradient, dim=[-2, -1])
    difference = np.linalg.norm((delta_torch - delta_manual).detach().numpy())
    assert difference < epsilon
