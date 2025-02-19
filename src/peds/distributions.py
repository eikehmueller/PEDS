import numpy as np
import scipy as sp
import petsc4py

petsc4py.init("-ksp_type preonly -pc_type lu")

from petsc4py import PETSc

__all__ = ["matern", "LogNormalDistribution1d", "LogNormalDistribution2d", "save_vtk"]


def matern(z, nu):
    """Matern covariance function

    1/(2^{nu-1}*Gamma(nu))*z^nu*K_nu(z)

    :arg z: value at which to evaluate
    :arg nu: index of function
    """
    if abs(nu - 1 / 2) < 1e-14:
        return np.exp(-np.abs(z))
    elif abs(nu - 3 / 2) < 1e-14:
        return (1 + np.abs(z)) * np.exp(-np.abs(z))
    elif abs(nu - 1) < 1e-14:
        return np.abs(z) * sp.special.kv(1, np.abs(z))
    else:
        return (
            0.5 ** (nu - 1)
            / sp.special.gamma(nu)
            * np.abs(z) ** nu
            * sp.special.kv(nu, np.abs(z))
        )


class LogNormalDistribution1d:
    """Draw samples alpha(x) for the field alpha with

    (-Delta + kappa^2)^{a/2} alpha(x) = W(x)

    on the unit interval where W(x) is white noise. kappa = 1/Lambda is
    the inverse correlation length and homogeneous Neumann BCs are
    assumed at the boundaries x=0, x=1. The generated field has Matern
    covariance with nu = a-d/2 where d=1 is the dimension.

    Reference:
      Lindgren, F., Rue, H. and Lindstroem, J., 2011. "An explicit link
      between Gaussian fields and Gaussian Markov random fields:
      the stochastic partial differential equation approach."
      Journal of the Royal Statistical Society Series B: Statistical
      Methodology, 73(4), pp.423-498.
    """

    def __init__(self, n, Lambda, a_power, seed=141517):
        """Initialise new instance

        :arg n: number of grid cells
        :arg Lambda: correlation length
        :arg a_power: power a
        :arg seed: seed of random number generator
        """
        super().__init__()
        assert a_power in (1, 2)
        self.n = n
        kappa = 1 / Lambda
        self.a_power = a_power
        self.rng = np.random.default_rng(seed=seed)
        h = 1.0 / self.n
        h_inv_sq = 1 / h**2

        Q_banded = np.empty((2, self.n + 1))
        Q_banded[1, :] = kappa**2 + 2 * h_inv_sq
        Q_banded[1, 0] = kappa**2 + h_inv_sq
        Q_banded[1, self.n] = kappa**2 + h_inv_sq
        Q_banded[0, :] = -h_inv_sq
        # banded Cholesky factor
        self.L_banded = sp.linalg.cholesky_banded(Q_banded, lower=False)
        nu = self.a_power - 1 / 2
        self.sigma2 = sp.special.gamma(nu) / (
            sp.special.gamma(nu + 1 / 2) * np.sqrt(4 * np.pi) * kappa ** (2 * nu)
        )

    def __iter__(self):
        """Iterator over dataset"""
        while True:
            xi = self.rng.normal(
                loc=0, scale=np.sqrt(self.n / self.sigma2), size=self.n + 1
            )
            if self.a_power == 2:
                yield sp.linalg.cho_solve_banded((self.L_banded, False), xi)
            else:
                yield sp.linalg.solve_banded((0, 1), self.L_banded, xi)


class LogNormalDistribution2d:
    """Draw samples alpha(x) for the field alpha with

    (-Delta + kappa^2) alpha(x) = W(x)

    on the unit square where W(x) is white noise. kappa = 1/Lambda is
    the inverse correlation length and homogeneous Neumann BCs are
    assumed at all boundaries. The generated field has Matern
    covariance with nu = 2-d/2 = 1 where d=2 is the dimension.
    """

    def __init__(self, n, Lambda, seed=141517):
        """Initialise new instance

        :arg n: number of grid cells
        :arg Lambda: correlation length
        :arg seed: seed of random number generator
        """
        super().__init__()
        self.n = n
        kappa = 1 / Lambda
        self.rng = np.random.default_rng(seed=seed)
        h = 1.0 / self.n
        h_inv_sq = 1 / h**2

        row_ptr = [0]
        col_indices = []
        values = []
        nnz = 0
        for j in range(self.n + 1):
            for k in range(self.n + 1):
                if j > 0:
                    K_left = h_inv_sq
                else:
                    K_left = 0
                if j < self.n:
                    K_right = h_inv_sq
                else:
                    K_right = 0
                if k > 0:
                    K_bottom = h_inv_sq
                else:
                    K_bottom = 0
                if k < self.n:
                    K_top = h_inv_sq
                else:
                    K_top = 0
                values.append(K_left + K_right + K_bottom + K_top + kappa**2)
                col_indices.append((self.n + 1) * j + k)
                nnz += 1
                if j > 0:
                    col_indices.append((self.n + 1) * (j - 1) + k)
                    values.append(-K_left)
                    nnz += 1
                if j < self.n:
                    col_indices.append((self.n + 1) * (j + 1) + k)
                    values.append(-K_right)
                    nnz += 1
                if k > 0:
                    col_indices.append((self.n + 1) * j + k - 1)
                    values.append(-K_bottom)
                    nnz += 1
                if k < self.n:
                    col_indices.append((self.n + 1) * j + k + 1)
                    values.append(-K_top)
                    nnz += 1
                row_ptr.append(nnz)
        # Create sparse matrix, KSP and PETSc solution vector
        self._petsc_mat = PETSc.Mat().createAIJ((self.n + 1) ** 2, nnz=nnz)
        self._petsc_mat.setValuesCSR(row_ptr, col_indices, values)
        self._petsc_mat.assemble()
        self._ksp = PETSc.KSP().create()
        self._ksp.setOperators(self._petsc_mat)
        self._ksp.setFromOptions()
        self.sigma2 = 1 / (4 * np.pi * kappa**2)
        self._alpha_vec = PETSc.Vec().createSeq((self.n + 1) ** 2)
        self._xi_vec = PETSc.Vec().createSeq((self.n + 1) ** 2)

    def __iter__(self):
        """Iterator over dataset"""
        while True:
            with self._xi_vec as v:
                v[:] = self.rng.normal(
                    loc=0, scale=self.n / np.sqrt(self.sigma2), size=(self.n + 1) ** 2
                )
            self._ksp.solve(self._xi_vec, self._alpha_vec)
            u = np.empty(shape=(self.n + 1, self.n + 1))
            with self._alpha_vec as v:
                u[:, :] = np.asarray(v[:]).reshape((self.n + 1, self.n + 1))
            yield u


def save_vtk(alpha_samples, filename):
    """Save a list of samples to vtk file

    :arg alpha_samples: list of fields
    :arg filename: name of file to write to
    """
    alpha = alpha_samples[0]
    nx = alpha.shape[0] - 1
    ny = alpha.shape[1] - 1
    npoints = (nx + 1) * (ny + 1)
    hx = 1.0 / nx
    hy = 1.0 / ny
    with open(filename, "w", encoding="utf8") as f:
        print("# vtk DataFile Version 2.0", file=f)
        print("Sample state", file=f)
        print("ASCII", file=f)
        print("DATASET STRUCTURED_POINTS", file=f)
        print(f"DIMENSIONS {nx + 1} {ny + 1} 1 ", file=f)
        print("ORIGIN -0.5 -0.5 0.0", file=f)
        print(f"SPACING {hx} {hy} 0", file=f)
        print(file=f)
        print(f"POINT_DATA {npoints}", file=f)
        for ell, alpha in enumerate(alpha_samples):
            print(f"SCALARS sample_{ell:03d} double 1", file=f)
            print("LOOKUP_TABLE default", file=f)
            for j in range(nx + 1):
                for k in range(nx + 1):
                    print(f"{alpha[j,k]:12.8e}", file=f)
