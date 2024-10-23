"""Test the diffusion model"""

import torch
import pytest
from peds.diffusion_model import tridiagonal_apply, tridiagonal_solve, DiffusionModel1d

torch.set_default_dtype(torch.float64)


@pytest.fixture
def rng():
    """Construct reproducible random number generator"""
    _rng = torch.Generator()
    _rng.manual_seed(25157)
    return _rng


def test_tridiagonal_solve_1d(rng):
    """Check that tridiagonal solve works for 1d tensors"""
    n = 32
    u_exact = torch.rand(n, generator=rng)
    alpha = torch.rand(n, generator=rng)
    K_diff = torch.exp(alpha)
    f_rhs = tridiagonal_apply(K_diff, u_exact)
    u = tridiagonal_solve(K_diff, f_rhs)
    error = (torch.norm(u - u_exact) / torch.norm(u_exact)).detach().numpy()
    tolerance = 1.0e-12
    assert error < tolerance


def test_tridiagonal_solve_batched(rng):
    """Check that tridiagonal solve works for batched tensors"""
    n = 32
    batchsize = 8
    m = 4
    u_exact = torch.rand((batchsize, m, n), generator=rng)
    alpha = torch.rand((batchsize, m, n), generator=rng)
    K_diff = torch.exp(alpha)
    f_rhs = tridiagonal_apply(K_diff, u_exact)
    u = tridiagonal_solve(K_diff, f_rhs)
    error = (torch.norm(u - u_exact) / torch.norm(u_exact)).detach().numpy()
    tolerance = 1.0e-12
    assert error < tolerance


def test_tridiagonal_solve_broadcast(rng):
    """Check that tridiagonal solve works for batched tensors if the RHS is a vector"""
    n = 32
    batchsize = 8
    m = 4
    f_rhs = torch.rand(n, generator=rng)
    alpha = torch.rand((batchsize, m, n), generator=rng)
    K_diff = torch.exp(alpha)
    u = tridiagonal_solve(K_diff, f_rhs)
    Au = tridiagonal_apply(K_diff, u)
    error = (torch.norm(Au - f_rhs) / torch.norm(f_rhs)).detach().numpy()
    tolerance = 1.0e-12
    assert error < tolerance


def test_gradient(rng):
    """Check that symbolic gradient matches finite difference approximation"""
    n = 32
    batchsize = 8
    m = 4
    alpha = torch.rand((batchsize, m, n), generator=rng, requires_grad=True)
    dalpha = torch.rand((batchsize, m, n), generator=rng)
    f_rhs = torch.rand(n, generator=rng)
    model = DiffusionModel1d(f_rhs)
    u = model(alpha)
    u_sq = torch.sum(u**2)
    u_sq.backward()
    grad_symbolic = torch.sum(alpha.grad * dalpha).detach().numpy()
    epsilon = 1.0e-8
    v = model(alpha + epsilon * dalpha)
    v_sq = torch.sum(v**2)
    grad_numerical = (v_sq - u_sq).detach().numpy() / epsilon
    error = abs(grad_symbolic - grad_numerical) / abs(grad_symbolic)
    tolerance = 1.0e-6
    assert error < tolerance
