import torch
import numpy as np
from peds.interpolation_1d import (
    VertexToVolumeInterpolator1d,
    VolumeToVertexInterpolator1d,
)


def test_vertex_to_volume_interpolation_1d():
    x = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    y_true = torch.tensor([[0.5, 1.5, 2.5, 3.5], [5.5, 6.5, 7.5, 8.5]])
    interpolator = VertexToVolumeInterpolator1d()
    y = interpolator(x)
    assert np.allclose((y - y_true).numpy(), 0)


def test_volume_to_vertex_interpolation_1d():
    x = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    y_true = torch.tensor([[0, 0.5, 1.5, 2.5, 3.5, 4], [5, 5.5, 6.5, 7.5, 8.5, 9]])
    interpolator = VolumeToVertexInterpolator1d()
    y = interpolator(x)
    assert np.allclose((y - y_true).numpy(), 0)
