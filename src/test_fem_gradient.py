"""Test computation of forward and backward evaluation of model problem

Let u=u(alpha) be defined by the solution of

    int_Omega exp(alpha) grad(u).grad(v) dx = int_Omega f v dx

for given alpha in V_alpha and all test functions v in V_u with homogeneous
Dirichlet boundary conditions on V_u.

The function construct_torch_operator() wraps this operation into a PyTorch
operator. Both the forward- and backward evaluation with the PyTorch operator
are compared to the manual computation.
"""

from firedrake import *
import numpy as np
import torch
import firedrake.ml.pytorch as fd_ml
import firedrake.adjoint as fda
import pytest


def construct_torch_operator(V_u, V_alpha, f_rhs):
    """Construct pytorch operator for the mapping alpha -> u = u(alpha)

    where u is the solution of the variational problem

        (grad(u), exp(alpha) * grad(v))_Omega = (v, f)_Omega

    for a given function alpha.

    :arg V_u: function space for the solution u,
        usually piecewise linear space CG_1)
    :arg V_alpha: function space for the coefficient field alpha,
        usually piecewise constant space DG_0
    :arg f_rhs: right hand side function f

    """
    fda.continue_annotation()  # record the following operations
    alpha = Function(V_alpha)
    phi = TestFunction(V_u)
    psi = TrialFunction(V_u)
    a_lhs = exp(alpha) * inner(grad(psi), grad(phi)) * dx
    b_rhs = phi * f_rhs * dx

    bcs = DirichletBC(V_u, 0, "on_boundary")
    u = Function(V_u)
    lvp = LinearVariationalProblem(a_lhs, b_rhs, u, bcs=bcs)
    lvs = LinearVariationalSolver(lvp)
    lvs.solve()
    fda.stop_annotating()
    return fd_ml.fem_operator(fda.ReducedFunctional(u, fda.Control(alpha)))


@pytest.fixture
def function_spaces():
    """Function spaces used for the discretisation"""
    nx = 16  # Number of grid cells in each direction
    mesh = UnitSquareMesh(nx, nx, quadrilateral=True)
    # CG function space of piecewise linear functions
    V = FunctionSpace(mesh, "CG", 1)
    # DG function space of piecewise constant functions
    V_DG = FunctionSpace(mesh, "DG", 0)
    return V, V_DG


@pytest.fixture
def f_rhs(function_spaces):
    """Right hand side function f"""
    V, _ = function_spaces
    x, y = SpatialCoordinate(V.mesh())
    return Function(V).interpolate(sin(pi * x) * cos(pi * y))


@pytest.fixture
def alpha(function_spaces):
    """Coefficient function alpha"""
    _, V_DG = function_spaces
    x, y = SpatialCoordinate(V_DG.mesh())
    return Function(V_DG).interpolate(-0.5 + x**2 + y**2)


@pytest.fixture
def bcs(function_spaces):
    """Coundary conditions"""
    V, _ = function_spaces
    return DirichletBC(V, 0, "on_boundary")


def test_forward(function_spaces, f_rhs, alpha, bcs):
    """Test forward evaluation

    :arg function_spaces: tuple of function spaces (V, V_DG)
    :arg f_rhs: right hand side function f
    :arg alpha: coefficient function alpha
    :arg bcs: boundary conditions
    """
    V, V_DG = function_spaces

    # Construct the PyTorch operator
    G = construct_torch_operator(V, V_DG, f_rhs)

    # Forward problem
    x_P = torch.tensor(alpha.dat.data, requires_grad=False)
    y_P = G(x_P)

    # Compute u by explicitly solving the weak problem
    phi = TestFunction(V)
    psi = TrialFunction(V)
    a_lhs = exp(alpha) * inner(grad(psi), grad(phi)) * dx
    b_rhs = phi * f_rhs * dx
    u = Function(V)
    solve(a_lhs == b_rhs, u, bcs=bcs)

    # relative error
    rel_error = np.linalg.norm(u.dat.data - y_P.numpy()) / np.linalg.norm(u.dat.data)
    tolerance = 1e-12
    assert rel_error < tolerance


def test_backward(function_spaces, f_rhs, alpha, bcs):
    """Test backward evaluation

    :arg function_spaces: tuple of function spaces (V, V_DG)
    :arg f_rhs: right hand side function f
    :arg alpha: coefficient function alpha
    :arg bcs: boundary conditions
    """
    V, V_DG = function_spaces
    mesh = V.mesh()
    phi = TestFunction(V)
    psi = TrialFunction(V)

    # Construct the PyTorch operator
    G = construct_torch_operator(V, V_DG, f_rhs)

    x_P = torch.tensor(alpha.dat.data, requires_grad=True)

    x, y = SpatialCoordinate(mesh)
    # Construct dual function
    g = Function(V).interpolate(cos(2 * pi * x) * cos(4 * pi * y))
    w_prime = assemble(phi * g * dx)
    y_P = G(x_P)
    w_P = torch.tensor(w_prime.dat.data).unsqueeze(0)
    y_P.backward(w_P)

    # Compute u by solving the weak problem
    a_lhs = inner(grad(psi), exp(alpha) * grad(phi)) * dx
    b_rhs = phi * f_rhs * dx

    u = Function(V)
    solve(a_lhs == b_rhs, u, bcs=bcs)

    # Compute adjoint solution lambda
    a_lhs_adjoint = exp(alpha) * inner(grad(psi), grad(phi)) * dx
    b_rhs_adjoint = phi * g * dx
    lmbda = Function(V)
    solve(a_lhs_adjoint == b_rhs_adjoint, lmbda, bcs=bcs)

    h_alpha = Function(V_DG).interpolate(cos(2 * pi * x) ** 2 * cos(pi * y) ** 2)

    z_1 = -assemble(exp(alpha) * h_alpha * inner(grad(u), grad(lmbda)) * dx)
    z_2 = np.dot(h_alpha.dat.data, x_P.grad.numpy())

    rel_grad_error = abs(z_1 - z_2) / abs(z_1)

    tolerance = 1e-12
    assert rel_grad_error < tolerance
