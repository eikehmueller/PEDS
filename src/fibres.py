import itertools
import numpy as np
from peds.auxilliary import save_vtk
from peds.distributions_fibres import FibreDistribution2d


n = 256
n_samples = 8
distribution = FibreDistribution2d(n, volume_fraction=0.55, gaussian_distribution=True)

samples = [np.exp(sample) for sample in list(itertools.islice(distribution, n_samples))]
save_vtk(samples, "fibre_samples.vtk")
