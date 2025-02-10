import numpy as np
from peds.diffusion_model_2d import Solver2d


def test_solver2d():
    rng = np.random.default_rng(seed=215725)
    m = 8
    f_rhs = rng.normal(size=(m, m))
    alpha = rng.normal(size=(m + 1, m + 1))
    solver = Solver2d(m)
    u = solver.solve(alpha, f_rhs)
