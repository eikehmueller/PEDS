import torch
import numpy as np

from peds.quantity_of_interest import QoISampling1d


def test_sampling1d():
    """Test 1d sampling QoI"""
    sample_points = [0.1, 0.25, 0.3, 0.8, 0.95]
    rng = torch.Generator().manual_seed(3265237)
    batch_size = 8
    m = 5
    n = 32
    u = torch.randn((batch_size, m, n), generator=rng)
    qoi = QoISampling1d(sample_points)
    v = qoi(u)
    idxs = [np.floor(x * n) for x in sample_points]
    assert np.allclose((v - u[..., idxs]).numpy(), 0)
