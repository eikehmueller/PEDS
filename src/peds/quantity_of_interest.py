import torch
import numpy as np

__all__ = ["QoISampling1d"]


class QoISampling1d(torch.nn.Module):
    """Quantity of interest that evaluates the field at a given set of 1d points"""

    def __init__(self, sample_points):
        """Initialise new instance

        :arg sample_points: points at which to evaluate the field
        """
        super().__init__()
        self.sample_points = np.asarray(sample_points)
        assert np.all(0 < self.sample_points) and np.all(self.sample_points < 1)

    @property
    def dim(self):
        """Dimension = number of sample points"""
        return self.sample_points.shape[0]

    def forward(self, x):
        n = x.shape[-1]
        idxs = np.floor(n * self.sample_points)
        return x[..., idxs]


class QoISampling2d(torch.nn.Module):
    """Quantity of interest that evaluates the field at a given set of 2d points"""

    def __init__(self, sample_points):
        """Initialise new instance

        :arg sample_points: points at which to evaluate the field
        """
        super().__init__()
        self.sample_points = np.asarray(sample_points)
        assert np.all(0 < self.sample_points) and np.all(self.sample_points < 1)

    @property
    def dim(self):
        """Dimension = number of sample points"""
        return self.sample_points.shape[0]

    def forward(self, x):
        N = np.asarray(x.shape[-2:])
        idxs = np.floor(N * self.sample_points).astype(int)
        return x[..., *idxs.T]
