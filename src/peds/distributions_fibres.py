import itertools
import numpy as np
from scipy.stats import norm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


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

    def draw(self, n_samples):
        """Draw given number of samples from distribution

        Returns a vector of fibre radii with the same length as the number of fibres.

        Depending on whether gaussian_distribution is True or False, the fibre radii are drawn from a
        normal distribution with given mean and variance, clipped to the
        range [self.min, self..max].
        """

        if self.gaussian:
            # number of nodal points for the CDF
            n_points = 1000
            nodal_points = np.linspace(self.r_min, self.r_max, num=n_points)
            pdf = norm.pdf(nodal_points, loc=self.r_avg, scale=self.sigma)

            cdf = np.cumsum(pdf)
            cdf /= cdf[-1]
            x = np.linspace(0, 1, n_points)
            # use inverse sampling transform
            xi = self._rng.uniform(low=0, high=1, size=n_samples)
            r_fibre = np.interp(xi, cdf, x) * (self.r_max - self.r_min) + self.r_min
        else:
            r_fibre = np.ones(shape=(n_samples,)) * self.r_avg
        return r_fibre


class FibreDistribution2d:
    """Artificial distribution of fibres in the domain

    Based on the Matlab code provided by Yang Chen.
    """

    def __init__(
        self,
        n,
        L=0.2,
        volume_fraction=0.55,
        r_fibre_dist=FibreRadiusDistribution(
            r_avg=7.5e-3, r_min=5.0e-3, r_max=10.0e-3, sigma=0.5e-4, gaussian=True
        ),
        kdiff_background=1.0,
        kdiff_fibre=0.01,
        seed=141517,
    ):
        """Initialise new instance

        :arg n: number of grid cells
        :arg L: side length of physical domain [in mm]
        :arg volume_fraction: volume fraction of fibres
        :arg r_fibre_dist: fibre radius distribution
        :arg kdiff_background: diffusion coefficient in background
        :arg kdiff_fibre: diffusion coefficient in fibre
        :arg seed: seed of random number generator
        """
        self.n = n
        self._volume_fraction = volume_fraction
        self._L = L
        self._r_fibre_dist = r_fibre_dist
        self._kdiff_background = kdiff_background
        self._kdiff_fibre = kdiff_fibre
        self._rng = np.random.default_rng(seed=seed)
        # Vertices of the grid onto which the diffusion coefficient is projected
        h = self._L / self.n
        X = np.arange(0, self._L + h / 2, h)
        self._vertices = np.asarray(
            [(y, x) for (x, y) in itertools.product(X, repeat=2)]
        ).reshape([len(X) ** 2, 2])
        # compute fibre positions
        n_fibres_per_direction = int(
            round(
                self._L
                / self._r_fibre_dist.r_avg
                * np.sqrt(self._volume_fraction / np.pi)
            )
        )
        # fibre diameter
        d_fibre = 2 * self._r_fibre_dist.r_avg
        X0 = (
            np.arange(0, (n_fibres_per_direction - 0.5) * d_fibre, d_fibre)
            * self._L
            / (d_fibre * n_fibres_per_direction)
        )
        self._fibre_locations = np.asarray(
            [p for p in itertools.product(X0, repeat=2)]
        ).reshape([len(X0) ** 2, 2])

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
            r_fibres = self._r_fibre_dist.draw(self._fibre_locations.shape[0])
            n_fibres = r_fibres.shape[0]
            labels = self._rng.permutation(range(n_fibres))
            eps_fibres = 3.0e-4
            it_max_ovlap = 5e3
            num_repeats = 30
            for _ in range(num_repeats):
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
            alpha = np.log(self._kdiff_background) * np.ones(
                shape=((self.n + 1) * (self.n + 1))
            )
            for p, r in zip(self._fibre_locations, r_fibres):
                dist = self._dist_periodic(p, self._vertices)
                alpha[np.where(dist < r)] = np.log(self._kdiff_fibre)
            alpha = np.reshape(alpha, (self.n + 1, self.n + 1))
            yield alpha
