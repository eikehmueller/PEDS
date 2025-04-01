import itertools
import numpy as np
import scipy as sp
import petsc4py
from scipy.stats import norm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

petsc4py.init("-ksp_type preonly -pc_type lu")

from petsc4py import PETSc

__all__ = [
    "matern",
    "LogNormalDistribution1d",
    "LogNormalDistribution2d",
    "FibreDistribution2d",
    "FibreDistributionYang2d",
    "save_vtk",
]


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


class FibreDistribution2d:
    """Artificial distribution of fibres in the domain

    The diffusion coefficient in the background material is fixed, and this is overlaid with
    a fixed number of fibres with a different diffusion coefficients, which is chosen (uniformly)
    from a given range. The thickness of the fibres is also drawn uniformly from a given range
    of thicknesses.
    """

    def __init__(
        self,
        n,
        n_fibres=8,
        d_fibre_min=0.03,
        d_fibre_max=0.05,
        kdiff_background=1.0,
        kdiff_fibre_min=0.01,
        kdiff_fibre_max=0.1,
        seed=141517,
    ):
        """

        :arg n: number of grid cells
        :arg nfibres: number of fibres per sample
        :arg d_fibre_min: minimal diameter of a single fibre
        :arg d_fibre_max: maximal diameter of a single fibre
        :arg kdiff_background: diffusion coefficient in background
        :arg kdiff_fibre_min: lower bound for diffusion coefficient in fibres
        :arg kdiff_fibre_max: upper bound for diffusion coefficient in fibres
        """
        self.n = n
        self._n_fibres = n_fibres
        self._d_fibre_min = d_fibre_min
        self._d_fibre_max = d_fibre_max
        self._kdiff_background = kdiff_background
        self._kdiff_fibre_min = kdiff_fibre_min
        self._kdiff_fibre_max = kdiff_fibre_max
        self._rng = np.random.default_rng(seed=seed)
        h = 1 / self.n
        X = np.arange(0, 1 + h / 2, h)
        Y = np.arange(0, 1 + h / 2, h)
        self._vertices = np.asarray(
            [(x, y) for y, x in itertools.product(Y, X)]
        ).reshape([self.n + 1, self.n + 1, 2])

    def __iter__(self):
        """Iterator over dataset"""

        while True:
            alpha = np.log(self._kdiff_background) * np.ones(
                shape=(self.n + 1, self.n + 1)
            )
            for _ in range(self._n_fibres):
                p0 = self._rng.uniform(low=0, high=1, size=[2])
                theta = self._rng.uniform(low=0, high=2 * np.pi)
                n0 = np.asarray([np.cos(theta), np.sin(theta)])
                d_fibre = self._rng.uniform(
                    low=self._d_fibre_min, high=self._d_fibre_max
                )
                k_diff = self._rng.uniform(
                    low=self._kdiff_fibre_min, high=self._kdiff_fibre_max
                )
                alpha_fibre = np.log(k_diff)
                alpha[np.where(np.abs((self._vertices - p0) @ n0) < d_fibre)] = (
                    alpha_fibre
                )

            yield alpha


class FibreDistributionYang2d:
    """Artificial distribution of fibres in the domain"""

    def __init__(
        self,
        n,
        volume_fraction=0.55,
        L=0.2,
        r_fibre_avg=7.5e-3,
        r_fibre_min=5.0e-3,
        r_fibre_max=10.0e-3,
        r_fibre_sigma=0.5e-4,
        gaussian_distribution=False,
        kdiff_background=1.0,
        kdiff_fibre=0.01,
        seed=141517,
    ):
        """

        :arg n: number of grid cells
        :arg volume_fraction: volume fraction of fibres
        :arg L: side length of physical domain [in mm]
        :arg r_fibre_avg: average fibre radius [in mm]
        :arg r_fibre_min: minimal fibre radius [in mm]
        :arg r_fibre_max: maximal fibre radius [in mm]
        :arg r_fibre_sigma: standard deviation of fibre radius [in mm]
        :arg gaussian_distribution: if True, draw fibre radii from a normal distribution, otherwise use set the
             radius of all fibres to r_fibre_avg
        :arg kdiff_background: diffusion coefficient in background
        :arg kdiff_fibre: diffusion coefficient in fibre
        :arg seed: seed of random number generator
        """
        self.n = n
        self._volume_fraction = volume_fraction
        self._L = L
        self._r_fibre_avg = r_fibre_avg
        self._r_fibre_min = r_fibre_min
        self._r_fibre_max = r_fibre_max
        self._r_fibre_sigma = r_fibre_sigma
        self._kdiff_background = kdiff_background
        self._kdiff_fibre = kdiff_fibre
        self._gaussian_distribution = gaussian_distribution
        self._rng = np.random.default_rng(seed=seed)
        # Vertices of the grid onto which the diffusion coefficient is projected
        h = self._L / self.n
        X = np.arange(0, self._L + h / 2, h)
        self._vertices = np.asarray(
            [p for p in itertools.product(X, repeat=2)]
        ).reshape([len(X) ** 2, 2])
        # compute fibre positions
        n_fibres_per_direction = int(
            round(self._L / self._r_fibre_avg * np.sqrt(self._volume_fraction / np.pi))
        )
        # fibre diameter
        d_fibre = 2 * self._r_fibre_avg
        X0 = (
            np.arange(0, (n_fibres_per_direction - 0.5) * d_fibre, d_fibre)
            * self._L
            / (d_fibre * n_fibres_per_direction)
        )
        self._fibre_locations = np.asarray(
            [p0 for p0 in itertools.product(X0, repeat=2)]
        ).reshape([len(X0) ** 2, 2])

    def _fibre_radii(self):
        """Draw fibre radii distribution

        Returns a vector of fibre radii with the same length as the number of fibres.

        Depending on whwther gaussian_distribution is True or False, the fibre radii are drawn from a
        normal distribution with given mean and variance, clipped to the range [r_fibre_min, r_fibre_max].
        """
        n_fibres = self._fibre_locations.shape[0]
        if self._gaussian_distribution:
            # number of nodal points for the CDF
            n_points = 1000
            nodal_points = np.linspace(
                self._r_fibre_min, self._r_fibre_max, num=n_points
            )
            pdf = norm.pdf(
                nodal_points, loc=self._r_fibre_avg, scale=self._r_fibre_sigma
            )

            cdf = np.cumsum(pdf)
            cdf /= cdf[-1]
            x = np.linspace(0, 1, n_points)
            # use inverse sampling transform
            xi = self._rng.uniform(low=0, high=1, size=n_fibres)
            r_fibre = (
                np.interp(xi, cdf, x) * (self._r_fibre_max - self._r_fibre_min)
                + self._r_fibre_min
            )
        else:
            r_fibre = np.ones(shape=(n_fibres,)) * self._r_fibre_avg
        return r_fibre

    def _dist_periodic(self, p, q):
        """Compute periodic distance between point p and array of points q

        For each q_j in q the periodic distance d_j is given by min_{offsets} |p+offset-q_j| for offsets
        in {-L, 0, +L}^2.

        Returns a vector d of distances d_j.

        :arg p: point in 2d, shape = (2,)
        :arg q: array of points in 2d, shape = (n, 2)
        """
        dist = np.empty(shape=(9, q.shape[0]))
        j = 0
        for j, offset in enumerate(
            itertools.product([-self._L, 0, +self._L], repeat=2)
        ):
            dist[j, :] = np.sqrt(np.sum((p + np.asarray(offset) - q) ** 2, axis=1))
        return np.min(dist, axis=0)

    def __iter__(self):
        """Iterator over dataset"""
        while True:
            r_fibres = self._fibre_radii()
            n_fibres = r_fibres.shape[0]
            labels = self._rng.permutation(range(n_fibres))
            eps_fibres = 3.0e-4
            it_max_ovlap = 5e3
            num_repeats = 30
            for k in range(num_repeats):
                tag_ovlap = 0
                for j in range(n_fibres):
                    # loop over fibres
                    p_j = self._fibre_locations[labels[j], :]
                    r_j = r_fibres[labels[j]]
                    dist = self._dist_periodic(p_j, self._fibre_locations)
                    dist[labels[j]] = np.inf
                    ur_sc = np.mean(sorted(dist)[:3])

                    dimin = 0
                    counter = 0
                    n_ovlap0 = np.inf
                    while dimin < eps_fibres:
                        u_r = self._rng.uniform(low=0, high=ur_sc)
                        u_theta = self._rng.uniform(low=0, high=2 * np.pi)
                        p_j_new = p_j + u_r * np.asarray(
                            [np.cos(u_theta), np.sin(u_theta)]
                        )
                        for dim in range(2):
                            while p_j_new[dim] < 0:
                                p_j_new[dim] += self._L
                            while p_j_new[dim] > self._L:
                                p_j_new[dim] -= self._L
                        dist = (
                            self._dist_periodic(p_j_new, self._fibre_locations)
                            - r_j
                            - r_fibres
                        )
                        dist[labels[j]] = np.inf
                        dimin = np.min(dist)
                        n_ovlap1 = np.count_nonzero(dist < 0)
                        if n_ovlap1 < n_ovlap0:
                            p_j_leastworst = p_j_new
                            n_ovlap0 = n_ovlap1
                        counter += 1
                        if counter > it_max_ovlap:
                            p_j_new = p_j_leastworst
                            tag_ovlap = tag_ovlap + 1
                            break
                    self._fibre_locations[labels[j], :] = p_j_new[:]
                if tag_ovlap == 0:
                    break

            self.visualise_fibres(
                self._fibre_locations, r_fibres, "fibre_positions.pdf"
            )

            alpha = np.log(self._kdiff_background) * np.ones(
                shape=((self.n + 1) * (self.n + 1))
            )
            for p, r in zip(self._fibre_locations, r_fibres):
                dist = self._dist_periodic(p, self._vertices)
                alpha[np.where(dist < r)] = np.log(self._kdiff_fibre)
            alpha = np.reshape(alpha, (self.n + 1, self.n + 1))
            yield alpha

    def visualise_fibres(self, fibre_locations, r_fibres, filename):
        """Auxilliary method to visualise the fibre distribution

        :arg fibre_locations: locations of the fibres
        :arg r_fibres: radii of the fibres
        :arg filename: name of the file to save the figure to
        """
        plt.clf()
        ax = plt.gca()
        ax.set_aspect("equal")
        d_fibre = 2 * np.min(r_fibres)
        ax.set_xlim(-d_fibre / 2, self._L + d_fibre / 2)
        ax.set_ylim(-d_fibre / 2, self._L + d_fibre / 2)
        ax.add_patch(
            mpatches.Rectangle(
                (0, 0), self._L, self._L, fill=False, edgecolor="black", lw=2
            )
        )
        for j in range(fibre_locations.shape[0]):
            for offset in itertools.product([-self._L, 0, +self._L], repeat=2):
                ax.add_patch(
                    mpatches.Circle(
                        fibre_locations[j] + np.asarray(offset),
                        r_fibres[j],
                        edgecolor="blue",
                        facecolor="blue",
                        fill=offset == (0, 0),
                    )
                )
        plt.savefig(filename)
        true_volume_fraction = np.pi * np.sum(r_fibres**2) / (self._L**2)
        print(
            f"Target / true volume fraction = {self._volume_fraction:.6f} / {true_volume_fraction:.6f}"
        )


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
