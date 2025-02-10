import torch
import numpy as np
import petsc4py

from petsc4py import PETSc

__all__ = ["Solver2d"]


class Solver2d:
    """Solver for 2d diffusion model"""

    def __init__(self, m):
        """Initialise new instance

        :arg m: number of grid cells in one dimension
        """
        self._m = m
        n = self._m**2
        self._row_ptr = [0]
        self._col_indices = []
        nnz = 0
        for j in range(self._m):
            for k in range(self._m):
                self._col_indices.append(self._m * j + k)
                nnz += 1
                if j > 0:
                    self._col_indices.append(self._m * (j - 1) + k)
                    nnz += 1
                if j < self._m - 1:
                    self._col_indices.append(self._m * (j + 1) + k)
                    nnz += 1
                if k > 0:
                    self._col_indices.append(self._m * j + k - 1)
                    nnz += 1
                if k < self._m - 1:
                    self._col_indices.append(self._m * j + k + 1)
                    nnz += 1
                self._row_ptr.append(nnz)
        self._petsc_mat = PETSc.Mat().createAIJ(n, nnz=nnz)
        self._ksp = PETSc.KSP().create()
        self._u_vec = PETSc.Vec().createSeq(n)

    def _set_matrix_values(self, alpha):
        """Set values in PETSc matrix"""
        values = np.empty(self._row_ptr[-1])
        nnz = 0
        # Construct PETSc matrix entries
        for j in range(self._m):
            for k in range(self._m):
                K_left = np.exp(0.5 * (alpha[j, k] + alpha[j, k + 1]))
                K_right = np.exp(0.5 * (alpha[j + 1, k] + alpha[j + 1, k + 1]))
                K_bottom = np.exp(0.5 * (alpha[j, k] + alpha[j + 1, k]))
                K_top = np.exp(0.5 * (alpha[j, k + 1] + alpha[j + 1, k + 1]))
                values[nnz] = K_left + K_right + K_bottom + K_top
                if j == 0:
                    values[nnz] += K_left
                nnz += 1
                if j > 0:
                    values[nnz] = -K_left
                    nnz += 1
                if j < self._m - 1:
                    values[nnz] = -K_right
                    nnz += 1
                if k > 0:
                    values[nnz] = -K_bottom
                    nnz += 1
                if k < self._m - 1:
                    values[nnz] = -K_top
                    nnz += 1
        self._petsc_mat.setValuesCSR(self._row_ptr, self._col_indices, values)
        self._petsc_mat.assemble()

    def solve(self, alpha, f_rhs):
        """Solve for a given alpha

        :arg alpha: diffusion field alpha, stored as a 2d tensor of shape (m+1,m+1)
        :arg rhs: given RHS
        """
        self._set_matrix_values(alpha)
        print("HERE")
        self._ksp.setOperators(self._petsc_mat)
        rhs_vec = PETSc.Vec().createWithArray(np.asarray(f_rhs).flatten())
        self._ksp.solve(rhs_vec, self._u_vec)
        u = np.empty((self._m, self._m))
        with self._u_vec as v:
            u[:, :] = np.asarray(v[:]).reshape((self._m, self._m))
        return u


class DiffusionModel2dOperator(torch.autograd.Function):
    """Differentiable function which solves the diffusion equation

    The diffusion equation is

        -div(K(x)grad(u)) = f(x)

    """

    def __init__(self):
        """Construct a new instance"""
        super().__init__()

    @staticmethod
    def forward(ctx, metadata, input):
        """Forward pass, compute u = u(alpha)

        :arg ctx: instance of object
        :arg metadata: metadata, contains information on the RHS
        :arg input: tensor containing alpha"""
        solver = metadata["solver"]
        f_rhs = metadata["f_rhs"]
        ctx.metadata.update(metadata)
        alpha = input.cpu()
        u = DiffusionModel2dOperator._solve(solver, alpha, f_rhs.cpu())
        ctx.save_for_backward(alpha, u)
        return u.to(input.device)

    @staticmethod
    def _solve(solver, alpha, f_rhs):
        if alpha.dim() == 2:
            u = torch.tensor(
                solver.solve(alpha.detach().numpy(), f_rhs.detach().numpy())
            )
        elif input.dim() > 2:
            batched_alpha = (
                torch.flatten(alpha, start_dim=0, end_dim=-3).detach().numpy()
            )
            batched_f_rhs = (
                torch.flatten(f_rhs, start_dim=0, end_dim=-3).detach().numpy()
            )
            batched_u = torch.empty_like(batched_f_rhs)
            u = torch.tensor
            for ell in range(batched_alpha.shape[0]):
                batched_u[ell, :, :] = solver.solve(
                    batched_alpha[ell, :, :], batched_f_rhs[ell, :, :]
                )
            u = torch.reshape(batched_u, f_rhs.shape)
        else:
            raise RuntimeError("tensor needs to be at least two-dimensional")
        return u

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass, compute dL/dalpha given dL/du

        :arg grad_output: dL/du
        """
        grad_input_shape = torch.Size(
            [
                *grad_output.shape[:-2],
                grad_output.shape[-2] + 1,
                grad_output.shape[-1] + 1,
            ]
        )
        alpha, u = ctx.saved_tensors
        solver = ctx.metadata["solver"]
        w = DiffusionModel2dOperator._solve(solver, alpha, grad_output.cpu())
        grad_input = torch.zeros(
            grad_input_shape, device=grad_output.device, dtype=grad_output.dtype
        )
        alpha = alpha.to(grad_input.device)
        for r in range(grad_output.shape[-2]):
            for s in range(grad_output.shape[-1]):
                # D^(1)_{rs}
                D_rs = 0.5 * (
                    torch.exp(0.5 * (alpha[..., r, s] + alpha[..., r, s + 1]))
                    + torch.exp(0.5 * (alpha[..., r, s] + alpha[..., r + 1, s]))
                )
                if r == 0:
                    D_rs += 0.5 * torch.exp(
                        0.5 * (alpha[..., r, s] + alpha[..., r, s + 1])
                    )
                grad_input[..., r, s] -= D_rs[...] * w[..., r, s] * u[..., r, s]
                # D^(2)_{rs}
                if r > 0:
                    D_rs = 0.5 * (
                        torch.exp(0.5 * (alpha[..., r, s] + alpha[..., r, s + 1]))
                        + torch.exp(0.5 * (alpha[..., r, s] + alpha[..., r - 1, s]))
                    )
                    grad_input[..., r, s] -= (
                        D_rs[...] * w[..., r - 1, s] * u[..., r - 1, s]
                    )
                # D^(3)_{rs}
                if s > 0:
                    D_rs = 0.5 * (
                        torch.exp(0.5 * (alpha[..., r, s] + alpha[..., r, s - 1]))
                        + torch.exp(0.5 * (alpha[..., r, s] + alpha[..., r + 1, s]))
                    )
                    if r > 0:
                        D_rs += 0.5 * torch.exp(
                            0.5 * (alpha[..., r, s] + alpha[..., r, s - 1])
                        )
                    grad_input[..., r, s] -= (
                        D_rs[...] * w[..., r, s - 1] * u[..., r, s - 1]
                    )
                # D^(4)_{rs}
                if r > 0 and s > 0:
                    D_rs = 0.5 * (
                        torch.exp(0.5 * (alpha[..., r, s] + alpha[..., r, s - 1]))
                        + torch.exp(0.5 * (alpha[..., r, s] + alpha[..., r - 1, s]))
                    )
                    grad_input[..., r, s] -= (
                        D_rs[...] * w[..., r - 1, s - 1] * u[..., r - 1, s - 1]
                    )
                # D^(5)_{rs}
                if r > 0:
                    D_rs = -0.5 * torch.exp(
                        0.5 * (alpha[..., r, s] + alpha[..., r, s + 1])
                    )
                    grad_input[..., r, s] -= D_rs[...] * (
                        w[..., r - 1, s] * u[..., r, s]
                        + w[..., r - 1, s - 1] * u[..., r, s - 1]
                    )
                # D^(6)_{rs}
                if s > 0:
                    D_rs = -0.5 * torch.exp(
                        0.5 * (alpha[..., r, s] + alpha[..., r + 1, s])
                    )
                    grad_input[..., r, s] -= D_rs[...] * (
                        w[..., r, s] * u[..., r - 1, s]
                        + w[..., r, s - 1] * u[..., r - 1, s - 1]
                    )
                # D^(7)_{rs}
                if r > 0:
                    D_rs = -0.5 * torch.exp(
                        0.5 * (alpha[..., r, s] + alpha[..., r, s - 1])
                    )
                    grad_input[..., r, s] -= D_rs[...] * (
                        w[..., r, s - 1] * u[..., r, s - 1]
                        + w[..., r, s - 1] * u[..., r - 1, s - 1]
                    )
                # D^(8)_{rs}
                if s > 0:
                    D_rs = -0.5 * torch.exp(
                        0.5 * (alpha[..., r, s] + alpha[..., r - 1, s])
                    )
                    grad_input[..., r, s] -= D_rs[...] * (
                        w[..., r, s] * u[..., r, s - 1]
                        + w[..., r - 1, s - 1] * u[..., r - 1, s - 1]
                    )
        return None, grad_output


class DiffusionModel2d(torch.nn.Module):

    def __init__(self, f_rhs):
        """Initialise a new instance

        :arg f_rhs: 1d tensor representing the right hand side"""
        super().__init__()
        m = f_rhs.shape[-1]
        solver = Solver2d(m)
        self.metadata = dict(solver=solver, f_rhs=torch.Tensor(f_rhs))

    def to(self, device):
        """Move to device

        :arg device: device to move to
        """
        new_self = super().to(device)
        new_self.metadata = dict(
            solver=self.metadata["solver"], f_rhs=self.metadata["f_rhs"].to(device)
        )
        return new_self

    def forward(self, x):
        """Apply model

        Computation is batched over all but the final dimension

        :arg x: diffusion tensor alpha, can be multidimensional"""
        return DiffusionModel2dOperator.apply(self.metadata, x)
