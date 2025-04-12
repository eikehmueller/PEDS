import torch
import numpy as np
import petsc4py

petsc4py.init("-ksp_type preonly -pc_type lu")

from petsc4py import PETSc

__all__ = ["Solver2d", "DiffusionModel2d"]


class Solver2d:
    """Solver for 2d diffusion model

    Provides PETSc KSP wrapper for solving the diffusion problem

        - div(exp(alpha(x,y)) grad(u(x,y))) = f_rhs(x,y)

    on a unit square with grid consisting of m x m cells. For a given right hand side
    (2d tensor of shape (m,m)) and field alpha (2d tensor of shape (m+1,m+1)) this computes
    the 2d tensor of shape (m,m) which describes the solution in grid cells. The field alpha is
    given at the vertices of the grid.
    """

    def __init__(self, m):
        """Initialise new instance

        :arg m: number of grid cells in one dimension
        """
        self._m = m
        self._hinv2 = m**2
        # Create row-pointer and column-index arrays
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
        # Create sparse matrix, KSP and PETSc solution vector
        n = self._m**2
        self._petsc_mat = PETSc.Mat().createAIJ(n, nnz=nnz)
        self._ksp = PETSc.KSP().create()
        self._ksp.setFromOptions()
        self._u_vec = PETSc.Vec().createSeq(n)

    def _set_matrix_values(self, alpha):
        """Set values in PETSc matrix

        :arg alpha: tensor of shape (m+1,m+1) which describes the field alpha at grid vertices
        """
        values = np.empty(self._row_ptr[-1])
        nnz = 0
        # Construct PETSc matrix entries
        for j in range(self._m):
            for k in range(self._m):
                # left flux
                K_left = np.exp(0.5 * (alpha[j, k] + alpha[j, k + 1]))
                # right flux
                if j < self._m - 1:
                    K_right = np.exp(0.5 * (alpha[j + 1, k] + alpha[j + 1, k + 1]))
                else:
                    K_right = 0
                # bottom flux
                if k > 0:
                    K_bottom = np.exp(0.5 * (alpha[j, k] + alpha[j + 1, k]))
                else:
                    K_bottom = 0
                # top flux
                if k < self._m - 1:
                    K_top = np.exp(0.5 * (alpha[j, k + 1] + alpha[j + 1, k + 1]))
                else:
                    K_top = 0
                # diagonal value
                values[nnz] = self._hinv2 * (K_left + K_right + K_bottom + K_top)
                if j == 0:
                    values[nnz] += self._hinv2 * K_left
                nnz += 1
                # left coupling
                if j > 0:
                    values[nnz] = -self._hinv2 * K_left
                    nnz += 1
                # right coupling
                if j < self._m - 1:
                    values[nnz] = -self._hinv2 * K_right
                    nnz += 1
                # bottom coupling
                if k > 0:
                    values[nnz] = -self._hinv2 * K_bottom
                    nnz += 1
                # top coupling
                if k < self._m - 1:
                    values[nnz] = -self._hinv2 * K_top
                    nnz += 1
        self._petsc_mat.setValuesCSR(self._row_ptr, self._col_indices, values)
        self._petsc_mat.assemble()

    def solve(self, alpha, f_rhs):
        """Solve for a given alpha and right hand side

        :arg alpha: diffusion field alpha, stored as a 2d tensor of shape (m+1,m+1)
        :arg rhs: given RHS, stored as a 2d tensor of shape (m,m)
        """
        self._set_matrix_values(alpha)
        self._ksp.setOperators(self._petsc_mat)
        rhs_vec = PETSc.Vec().createWithArray(np.asarray(f_rhs).flatten())
        self._ksp.solve(rhs_vec, self._u_vec)
        u = np.empty((self._m, self._m))
        with self._u_vec as v:
            u[:, :] = np.asarray(v[:]).reshape((self._m, self._m))
        return u

    def apply_operator(self, alpha, u):
        """Apply discretised operator

        :arg alpha: diffusion tensor, stored as a 2d tensor of shape (m+1,m+1)
        :arg u: field to which the operator is applied, stored as a 2d tensor of shape (m,m)
        """
        v = np.zeros_like(u)
        for j in range(self._m):
            for k in range(self._m):
                # left flux
                K_left = np.exp(0.5 * (alpha[j, k] + alpha[j, k + 1]))
                # right flux
                if j < self._m - 1:
                    K_right = np.exp(0.5 * (alpha[j + 1, k] + alpha[j + 1, k + 1]))
                else:
                    K_right = 0
                # bottom flux
                if k > 0:
                    K_bottom = np.exp(0.5 * (alpha[j, k] + alpha[j + 1, k]))
                else:
                    K_bottom = 0
                # top flux
                if k < self._m - 1:
                    K_top = np.exp(0.5 * (alpha[j, k + 1] + alpha[j + 1, k + 1]))
                else:
                    K_top = 0
                # diagonal value
                v[j, k] += self._hinv2 * (K_left + K_right + K_bottom + K_top) * u[j, k]
                if j == 0:
                    v[j, k] += self._hinv2 * K_left * u[j, k]
                # left coupling
                if j > 0:
                    v[j, k] -= self._hinv2 * K_left * u[j - 1, k]
                # right coupling
                if j < self._m - 1:
                    v[j, k] -= self._hinv2 * K_right * u[j + 1, k]
                # bottom coupling
                if k > 0:
                    v[j, k] -= self._hinv2 * K_bottom * u[j, k - 1]
                # top coupling
                if k < self._m - 1:
                    v[j, k] -= self._hinv2 * K_top * u[j, k + 1]
        return v


class DiffusionModel2dOperator(torch.autograd.Function):
    """Differentiable function which solves the diffusion equation

    The diffusion equation is

        -div(exp(alpha(x,y)grad(u(x,y))) = f(x,y)

    and this class provides a differentiable operator from alpha (a 2d tensor of shape
    (a,m+1,m+1) with 'a' standing for batch dimensions) to u (a 2d tensor of shape (a,m,m)).
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
        batch_dims = input.shape[:-2]
        u = DiffusionModel2dOperator._solve(
            solver, input.cpu(), f_rhs.cpu().expand(*batch_dims, -1, -1)
        ).to(input.device)
        ctx.save_for_backward(input, u)
        return u

    @staticmethod
    def _solve(solver, alpha, f_rhs):
        """Batched solve

        Solve the linear system A u = f for given alpha  and right hand side f

        :arg solver: PETSc solver wrapper class
        :arg alpha: diffusion tensor alpha, shape = (a,m+1,m+1)
        :arg f_rhs: right hand side f, shape = (a,m,m)
        """
        if alpha.dim() == 2:
            # if there are no batch-dimensions simply solve
            u = solver.solve(alpha.detach().numpy(), f_rhs.detach().numpy())

        elif alpha.dim() > 2:
            # otherwise, flatten batch dimensions and solve in each of them
            batched_alpha = (
                torch.flatten(alpha, start_dim=0, end_dim=-3).detach().numpy()
            )
            batched_f_rhs = (
                torch.flatten(f_rhs, start_dim=0, end_dim=-3).detach().numpy()
            )
            batched_u = np.empty_like(batched_f_rhs)
            u = torch.tensor
            for ell in range(batched_alpha.shape[0]):
                batched_u[ell, :, :] = solver.solve(
                    batched_alpha[ell, :, :], batched_f_rhs[ell, :, :]
                )
            u = batched_u.reshape(f_rhs.shape)
        else:
            raise RuntimeError("tensor needs to be at least two-dimensional")
        return torch.tensor(u, dtype=f_rhs.dtype, device=f_rhs.device)

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
        w = DiffusionModel2dOperator._solve(solver, alpha.cpu(), grad_output.cpu()).to(
            grad_output.device
        )
        grad_input = torch.zeros(
            grad_input_shape, device=grad_output.device, dtype=grad_output.dtype
        )
        m = grad_output.shape[-1]
        h_inv2 = m**2
        for r in range(m + 1):
            for s in range(m + 1):
                if (1 <= r) and (r <= m - 1) and (1 <= s):
                    F_x = (
                        0.5
                        * h_inv2
                        * torch.exp(0.5 * (alpha[..., r, s - 1] + alpha[..., r, s]))
                    )
                    z_x = (
                        w[..., r, s - 1] * u[..., r, s - 1]
                        + w[..., r - 1, s - 1] * u[..., r - 1, s - 1]
                        - w[..., r, s - 1] * u[..., r - 1, s - 1]
                        - w[..., r - 1, s - 1] * u[..., r, s - 1]
                    )
                    grad_input[..., r, s] -= F_x * z_x
                    grad_input[..., r, s - 1] -= F_x * z_x
                if (1 <= r) and (1 <= s) and (s <= m - 1):
                    F_y = (
                        0.5
                        * h_inv2
                        * torch.exp(0.5 * (alpha[..., r - 1, s] + alpha[..., r, s]))
                    )
                    z_y = (
                        w[..., r - 1, s] * u[..., r - 1, s]
                        + w[..., r - 1, s - 1] * u[..., r - 1, s - 1]
                        - w[..., r - 1, s] * u[..., r - 1, s - 1]
                        - w[..., r - 1, s - 1] * u[..., r - 1, s]
                    )
                    grad_input[..., r, s] -= F_y * z_y
                    grad_input[..., r - 1, s] -= F_y * z_y
                if (r == 0) and (1 <= s) and (s <= m):
                    F_y0 = (
                        0.5
                        * h_inv2
                        * torch.exp(0.5 * (alpha[..., r, s - 1] + alpha[..., r, s]))
                    )
                    z_y = w[..., r, s - 1] * u[..., r, s - 1]
                    grad_input[..., r, s] -= 2 * F_y0 * z_y
                    grad_input[..., r, s - 1] -= 2 * F_y0 * z_y
        return None, grad_input


class DiffusionModel2d(torch.nn.Module):

    def __init__(self, f_rhs):
        """Initialise a new instance

        :arg f_rhs: 2d tensor representing the right hand side"""
        super().__init__()
        m = f_rhs.shape[-1]
        solver = Solver2d(m)
        self.metadata = dict(solver=solver, f_rhs=torch.Tensor(f_rhs))

    def to(self, device):
        """Move to device

        :arg device: device to move to
        """
        super().to(device)
        self.metadata = dict(
            solver=self.metadata["solver"], f_rhs=self.metadata["f_rhs"].to(device)
        )
        return self

    def forward(self, x):
        """Apply model

        Computation is batched over all but the final dimension

        :arg x: diffusion tensor alpha, can be multidimensional"""
        return DiffusionModel2dOperator.apply(self.metadata, x)

    def coarsen(self, scaling_factor):
        """Return a coarsened version of the model

        :arg scaling_factor: coarsening factor, must be an integer divisor of problem size
        """
        f_rhs = self.metadata["f_rhs"]
        n = f_rhs.shape[-1]
        assert (
            n == (n // scaling_factor) * scaling_factor
        ), "scaling factor must divide problem size"
        # Construct coarse RHS by averaging over final two dimensions
        f_rhs_coarse = torch.squeeze(
            torch.nn.functional.avg_pool2d(
                torch.unsqueeze(torch.unsqueeze(f_rhs, 0), 0),
                kernel_size=(scaling_factor, scaling_factor),
            )
        )
        return DiffusionModel2d(f_rhs_coarse)
