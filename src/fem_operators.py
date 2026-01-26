import numpy as np
import torch
from firedrake import *
from firedrake.__future__ import interpolate

import firedrake.ml.pytorch as fd_ml
import firedrake.adjoint as fda


class BatchedOperator(torch.nn.Module):
    """Batch an operator

    Given an operator which maps x of shape (n,) or (1,n) to y of shape (1,m),
    construct an operator which maps either:

        1. x (shape (n,)) -> y (shape (m,))
        2. x (shape (batchsize,n)) -> y (shape (batchsize,m))

    :arg op: non-batched operator
    """

    def __init__(self, op):
        super().__init__()
        self._op = op

    def forward(self, x):
        """Batched apply of operator

        :arg x: tensor of shape (n,) or (batchsize,n)
        """
        # Add batch dimension if x is of shape (n,)
        if x.ndim == 1:
            _x = torch.unsqueeze(x, 0)
        else:
            _x = x
        # Apply operator to each sample
        y = torch.stack(
            [torch.squeeze(self._op(z)) for z in torch.unbind(_x, dim=0)], dim=0
        )
        # Remove batch dimension if x is of shape (n,)
        if x.ndim == 1:
            return y.squeeze()
        else:
            return y


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
    return BatchedOperator(
        fd_ml.fem_operator(fda.ReducedFunctional(u, fda.Control(alpha)))
    )


def construct_qoi(V_u, points):
    """Construct pytorch operator for the mapping u -> qoi(u)

    where qoi is the evaluation of u at a set uf points

    :arg V_u: function space for the solution u,
        usually piecewise linear space CG_1)
    :arg points: List of point, must be shape (n,dim) where dim is the
        dimension of the mesh
    """

    fda.continue_annotation()  # record the following operations
    vertex_only_mesh = VertexOnlyMesh(V_u.mesh(), points)
    u = Function(V_u)
    V_qoi = FunctionSpace(vertex_only_mesh, "DG", 0)
    u_at_points = assemble(interpolate(u, V_qoi))
    fda.stop_annotating()
    return BatchedOperator(
        fd_ml.fem_operator(fda.ReducedFunctional(u_at_points, fda.Control(u)))
    )


batchsize = 8  # Batch size
nx = 16  # Number of grid cells in each direction
points = [[0.1], [0.3], [0.7]]
npoints = len(points)

rng = np.random.default_rng(seed=7215563)

mesh = UnitIntervalMesh(nx)
(x,) = SpatialCoordinate(mesh)
# CG function space of piecewise linear functions
V = FunctionSpace(mesh, "CG", 1)
# DG function space of piecewise constant functions
V_DG = FunctionSpace(mesh, "DG", 0)

f_rhs = Function(V).interpolate(x**2)

solver = construct_torch_operator(V, V_DG, f_rhs)
qoi = construct_qoi(V, points)
loss = torch.nn.MSELoss()

alpha = torch.tensor(rng.normal(size=(batchsize, nx)))
q_true = torch.tensor(rng.normal(size=(batchsize, npoints)))

u = solver(alpha)
q = qoi(u)
output = loss(q, q_true)

print(output)
