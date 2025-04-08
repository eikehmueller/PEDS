"""Distributions that represent artificial distribution of fibres in the domain

This module contains classes that allow to generate a distribution of fibres in 2D and
a distribution of fibre radii. The distribution of fibre radii is a (clipped) normal distribution.

Reference for the algorithm:
    Yang Chen (2022): "Towards a data fusion framework: combining multi-modal data for composites
    certification" Technical report, DETI-PoC2-003


"""

import os
import tempfile
import ctypes
import subprocess

import itertools
import numpy as np
from scipy.stats import norm


__all__ = ["FibreRadiusDistribution", "FibreDistribution2d"]


class FibreRadiusDistribution:
    """Class representing the distribution of fibre radii"""

    def __init__(self, r_avg, r_min, r_max, sigma, gaussian, seed=856219):
        """Initialise new instance

        :arg avg: average fibre radius
        :arg min: minimum fibre radius
        :arg max: maximum fibre radius
        :arg sigma: standard deviation of fibre radius
        :arg gaussian: whether to draw from a gaussian distribution
        """
        self.r_avg = r_avg
        self.r_min = r_min
        self.r_max = r_max
        self.sigma = sigma
        self.gaussian = gaussian
        self._rng = np.random.default_rng(seed=seed)
        # number of nodal points for the CDF
        n_points = 1000
        nodal_points = np.linspace(self.r_min, self.r_max, num=n_points)
        pdf = norm.pdf(nodal_points, loc=self.r_avg, scale=self.sigma)

        self._cdf = np.cumsum(pdf)
        self._cdf /= self._cdf[-1]
        self._x_cdf = np.linspace(0, 1, n_points)

    def draw(self, n_samples):
        """Draw given number of samples from distribution

        Returns a vector of fibre radii with the same length as the number of fibres.

        Depending on whether gaussian_distribution is True or False, the fibre radii are drawn from a
        normal distribution with given mean and variance, clipped to the
        range [self.min, self..max].
        """

        if self.gaussian:
            # use inverse sampling transform
            xi = self._rng.uniform(low=0, high=1, size=n_samples)
            r_fibre = (
                np.interp(xi, self._cdf, self._x_cdf) * (self.r_max - self.r_min)
                + self.r_min
            )
        else:
            r_fibre = np.ones(shape=(n_samples,)) * self.r_avg
        return r_fibre


class FibreDistribution2d:
    """Artificial distribution of fibres in the domain

    Based on the Matlab code provided by Yang Chen, Fig. 15 in the above reference.
    """

    def __init__(
        self,
        n,
        volume_fraction=0.55,
        r_fibre_dist=FibreRadiusDistribution(
            r_avg=7.5e-3, r_min=5.0e-3, r_max=10.0e-3, sigma=0.5e-3, gaussian=True
        ),
        kdiff_background=1.0,
        kdiff_fibre=0.1,
        seed=141517,
        fast_code=True,
    ):
        """Initialise new instance

        :arg n: number of grid cells
        :arg volume_fraction: volume fraction of fibres
        :arg r_fibre_dist: fibre radius distribution, instance of class FibreRadiusDistribution
        :arg kdiff_background: diffusion coefficient in background
        :arg kdiff_fibre: diffusion coefficient in fibre
        :arg seed: seed of random number generator
        :arg fast_code: use generated C code
        """
        self.n = n
        self._volume_fraction = volume_fraction
        self._r_fibre_dist = r_fibre_dist
        self._kdiff_background = kdiff_background
        self._kdiff_fibre = kdiff_fibre
        self._rng = np.random.default_rng(seed=seed)
        # Vertices of the grid onto which the diffusion coefficient is projected
        h = 1 / self.n
        X = np.arange(0, 1 + h / 2, h)
        self._vertices = np.asarray(
            [(y, x) for (x, y) in itertools.product(X, repeat=2)]
        ).reshape([len(X) ** 2, 2])
        # compute initial fibre locations, arranged in a regular grid
        n_fibres_per_direction = int(
            round(1 / self._r_fibre_dist.r_avg * np.sqrt(self._volume_fraction / np.pi))
        )
        # fibre diameter
        d_fibre = 2 * self._r_fibre_dist.r_avg
        X0 = np.arange(0, (n_fibres_per_direction - 0.5) * d_fibre, d_fibre) / (
            d_fibre * n_fibres_per_direction
        )
        self._initial_fibre_locations = np.asarray(
            [p for p in itertools.product(X0, repeat=2)]
        ).reshape([len(X0) ** 2, 2])
        self._fast_code = fast_code
        if self._fast_code:
            lib_path, _ = os.path.split(os.path.realpath(__file__))
            source_file = os.path.join(lib_path, "libfibres.cc")
            with tempfile.TemporaryDirectory() as tmp_dir:
                lib_file = os.path.join(tmp_dir, "libfibres.so")
                try:
                    subprocess.run(
                        ["g++", "-fPIC", "-shared", "-O3", "-o", lib_file, source_file],
                        check=True,
                    )
                except subprocess.CalledProcessError as e:
                    print("Error compiling C++ code for fibre distribution: " + str(e))
                    print("Falling back to Python implementation.")
                    self._fast_code = False
                if self._fast_code:
                    # Load the shared library
                    self._fast_dist_periodic = ctypes.CDLL(lib_file).dist_periodic
                    self._fast_dist_periodic.argtypes = [
                        ctypes.c_int,
                        np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                    ]

                    initialise_rng = ctypes.CDLL(lib_file).initialise_rng
                    initialise_rng.argtypes = [
                        ctypes.c_uint,
                        ctypes.POINTER(ctypes.c_uint),
                        ctypes.c_char_p,
                    ]

                    self._fast_move_fibres = ctypes.CDLL(lib_file).move_fibres
                    self._fast_move_fibres.argtypes = [
                        ctypes.c_uint,
                        np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ctypes.c_uint,
                        ctypes.c_double,
                        ctypes.c_uint,
                        ctypes.POINTER(ctypes.c_uint),
                        ctypes.c_char_p,
                    ]

                    seed = 735283
                    self._rng_state_length = ctypes.c_uint()
                    self._rng_state = ctypes.c_char_p()
                    self._rng_state.value = 10000 * b" "

                    initialise_rng(
                        seed, ctypes.byref(self._rng_state_length), self._rng_state
                    )

    def __iter__(self):
        """Iterator over dataset"""
        num_repeats = 30
        it_max_ovlap = 5000
        eps_fibres = 0.0015
        while True:
            r_fibres = self._r_fibre_dist.draw(self._initial_fibre_locations.shape[0])
            fibre_locations = np.array(self._initial_fibre_locations)
            if self._fast_code:
                n_fibres = r_fibres.shape[0]
                self._fast_move_fibres(
                    n_fibres,
                    r_fibres,
                    fibre_locations,
                    num_repeats,
                    eps_fibres,
                    it_max_ovlap,
                    self._rng_state_length,
                    self._rng_state,
                )
            else:
                fibre_locations = self._move_fibres(
                    fibre_locations,
                    r_fibres,
                    num_repeats=30,
                    it_max_ovlap=5000,
                    eps_fibres=0.0015,
                )
            # Project onto grid
            alpha = np.log(self._kdiff_background) * np.ones(
                shape=((self.n + 1) * (self.n + 1))
            )
            for p, r in zip(fibre_locations, r_fibres):
                dist = self._dist_periodic(p, self._vertices)
                alpha[np.where(dist < r)] = np.log(self._kdiff_fibre)
            alpha = np.reshape(alpha, (self.n + 1, self.n + 1))
            yield alpha

    def _move_fibres(
        self,
        fibre_locations,
        r_fibres,
        num_repeats=30,
        it_max_ovlap=5000,
        eps_fibres=0.0015,
    ):
        """Randomly move the fibres according to the algorithm by Yang Chen

        Returns the new fibre locations.

        :arg initial_fibre_locations: initial locations of the fibres
        :arg r_fibres: fibre radii
        :arg num_repeats: number of repetitions
        :arg it_max_ovlap: maximum number of iterations to resolve overlap
        :arg eps_fibres: minimum distance between fibres
        """
        n_fibres = r_fibres.shape[0]
        labels = self._rng.permutation(range(n_fibres))
        overlap = True
        k = 0
        while overlap or k < num_repeats:
            overlap = False
            for j in range(n_fibres):
                # loop over fibres
                p_j = fibre_locations[labels[j], :]
                r_j = r_fibres[labels[j]]
                dist = self._dist_periodic(p_j, fibre_locations)
                dist[labels[j]] = np.inf
                ur_sc = np.mean(sorted(dist)[:3])

                dimin = 0
                counter = 0
                n_ovlap0 = np.inf
                while dimin < eps_fibres:
                    u_r = self._rng.uniform(low=0, high=ur_sc)
                    u_theta = self._rng.uniform(low=0, high=2 * np.pi)
                    p_j_new = p_j + u_r * np.asarray([np.cos(u_theta), np.sin(u_theta)])
                    for dim in range(2):
                        while p_j_new[dim] < 0:
                            p_j_new[dim] += 1
                        while p_j_new[dim] > 1:
                            p_j_new[dim] -= 1
                    dist = (
                        self._dist_periodic(p_j_new, fibre_locations) - r_j - r_fibres
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
                        overlap = True
                        break
                fibre_locations[labels[j], :] = p_j_new[:]
            k += 1
        return fibre_locations

    def _dist_periodic(self, p, q):
        """Compute periodic distance between point p and array of points q

        For each q_j in q the periodic distance d_j is given by min_{offsets} |p+offset-q_j| for offsets
        in {-L, 0, +L}^2.

        Returns a vector d of distances d_j.

        :arg p: point in 2d, shape = (2,)
        :arg q: array of points in 2d, shape = (n, 2)
        """

        n = q.shape[0]
        if self._fast_code:
            min_dist = np.empty(n, dtype=np.float64)
            extents = np.asarray([1.0, 1.0], dtype=np.float64)
            self._fast_dist_periodic(n, p, q, extents, min_dist)
            return min_dist
        else:
            j = 0
            dist = np.empty(shape=(9, n))
            for j, offset in enumerate(itertools.product([-1, 0, +1], repeat=2)):
                dist[j, :] = np.sqrt(np.sum((p + np.asarray(offset) - q) ** 2, axis=1))
            return np.min(dist, axis=0)
