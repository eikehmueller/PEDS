"""Provide functionality for interpolating a function that is defined on the vertices
of a 1d grid to a function on the volumes and vice-versa"""

import torch

__all__ = ["VertexToVolumeInterpolator1d", "VolumeToVertexInterpolator1d"]


class VertexToVolumeInterpolator1d(torch.nn.Module):
    """Interpolate vertex data to volumes

    If x is a tensor with final shape n+1 we have that

        y_{a,j} = 1/2*(x_{a,j}+x_{a,j+1})    for j=0,1,...,n-1

    where a stands for all batch indices. y is a tensor with final shape n.

    """

    def __init__(self):
        """Initialise new instance"""
        super().__init__()

    def forward(self, x):
        """Forward evaluation"""
        return 0.5 * (x[..., :-1] + x[..., 1:])


class VolumeToVertexInterpolator1d(torch.nn.Module):
    """Interpolate volume data to vertices

    If x is a tensor with final shape n we have that:

                  { x_{a,j}                    for j=0
        y_{a,j} = { 1/2*(x_{a,j}+x_{a,j-1})    for j=1,2,...,n-1
                  { x_{a,n-1}                  for j=n

    where a stands for all batch indices. y is a tensor with final shape n+1

    """

    def __init__(self):
        """Initialise new instance"""
        super().__init__()

    def forward(self, x):
        """Forward evaluation"""
        y = torch.empty(size=x.shape[:-1] + (x.shape[-1] + 1,), device=x.device)
        y[..., 0] = x[..., 0]
        y[..., 1:-1] = 0.5 * (x[..., :-1] + x[..., 1:])
        y[..., -1] = x[..., -1]
        return y
