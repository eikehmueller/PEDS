import itertools
import torch
import numpy as np
from peds.auxilliary import save_vtk
from peds.distributions import LogNormalDistribution2d
from peds.interpolation_2d import (
    VertexToVolumeInterpolator2d,
    VolumeToVertexInterpolator2d,
)
from firedrake import *


n = 128
Lambda = 0.1  # correlation length

distribution = LogNormalDistribution2d(n, Lambda)

n_samples = 16
scaling_factor = 4

alpha_samples = list(itertools.islice(iter(distribution), n_samples))

alpha = torch.tensor(alpha_samples)

downsampler = torch.nn.Sequential(
    torch.nn.Unflatten(-2, (1, n + 1)),
    VertexToVolumeInterpolator2d(),
    torch.nn.AvgPool2d(1, stride=scaling_factor),
    VolumeToVertexInterpolator2d(),
    torch.nn.Flatten(-3, -2),
)

alpha_coarse = downsampler(alpha)

alpha_coarse_samples = []
for j in range(n_samples):
    alpha_coarse_samples.append(alpha_coarse[j, :, :].numpy())


save_vtk(alpha_samples, "samples.vtk")
save_vtk(alpha_coarse_samples, "samples_coarse.vtk")
