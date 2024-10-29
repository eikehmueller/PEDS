import torch
import numpy as np

__all__ = ["QoISampling1d"]


class QoISampling1d(torch.nn.Module):
    """Quantity of interest that evaluates the field at a given set of points"""

    def __init__(self, sample_points):
        """Initialise new instance

        :arg sample_points: points at which to evaluate the field
        """
        super().__init__()
        self.sample_points = np.asarray(sample_points)
        assert np.all(0 < self.sample_points) and np.all(self.sample_points < 1)

    def forward(self, x):
        n = x.shape[-1]
        idxs = np.floor(n * self.sample_points)
        return x[..., idxs]
