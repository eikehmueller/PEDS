import itertools
import numpy as np
from peds.auxilliary import save_vtk
from peds.distributions_fibres import FibreDistribution2d, FibreRadiusDistribution


n = 256
n_samples = 8
distribution = FibreDistribution2d(
    n,
    volume_fraction=0.55,
    r_fibre_dist=FibreRadiusDistribution(
        avg=7.5e-3, min=5.0e-3, max=10.0e-3, sigma=0.5e-4, gaussian=True
    ),
)

samples = [np.exp(sample) for sample in list(itertools.islice(distribution, n_samples))]
save_vtk(samples, "fibre_samples.vtk")
