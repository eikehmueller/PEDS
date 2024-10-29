import numpy as np
import scipy as sp

__all__ = ["matern", "LogNormalDistribution1d"]


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
    else:
        return (
            0.5 ** (nu - 1)
            / sp.special.gamma(nu)
            * np.abs(z) ** nu
            * sp.special.kv(nu, np.abs(z))
        )


class LogNormalDistribution1d:
    """Draw samples alpha(x) for the field alpha with

    (-Delta + kappa^2)^a alpha(x) = W(x)

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
