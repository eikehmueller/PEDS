"""Provide functionality for interpolating a function that is defined on the vertices
of a 2d grid to a function on the volumes and vice-versa"""

import torch

__all__ = ["VertexToVolumeInterpolator2d", "VolumeToVertexInterpolator2d"]


class VertexToVolumeInterpolator2d(torch.nn.Module):
    """Interpolate vertex data to volumes

    If x is a tensor with final shape (nx+1,ny+1) we have that

        y_{a,i,j} = 1/4*(x_{a,i,j}+x_{a,i+1,j}+x_{a,i,j+1}+x_{a,i+1,j+1})

           for i=0,1,...,nx-1, j=0,1,...,ny-1

    where a stands for all batch indices. y is a tensor with final shape (nx,ny).

    """

    def __init__(self):
        """Initialise new instance"""
        super().__init__()

    def forward(self, x):
        """Forward evaluation"""
        return 0.25 * (
            x[..., :-1, :-1] + x[..., 1:, :-1] + x[..., :-1, 1:] + x[..., 1:, 1:]
        )


class VolumeToVertexInterpolator2d(torch.nn.Module):
    """Interpolate volume data to vertices

    If x is a tensor with final shape (nx,ny) we have that:

                    { x_{a,0,0}                       for i=0, j=0
                    { x_{a,nx-1,0}                    for i=nx, j=0
                    { x_{a,0,ny-1}                    for i=0, j=ny
                    { x_{a,nx-1,ny-1}                 for i=nx, j=ny
        y_{a,i,j} = { 1/4*( x_{a,i,j}+x_{a,i-1,j}        for i=1,2,...,nx-1
                    {     + x_{a,i,j-1}+x_{a,i-1,j-1} )    and j=1,2,...,ny-1
                    { 1/2*(x_{a,i,0}+x_{a,i-1,0})       for i=1,2,...,nx, j=0
                    { 1/2*(x_{a,i,ny-1}+x_{a,i-1,ny-1}) for i=1,2,...,nx, j=ny
                    { 1/2*(x_{a,0,j}+x_{a,0,j-1})       for i=0, j=1,2,...,ny
                    { 1/2*(x_{a,nx-1,j}+x_{a,nx-1,j-1}) for i=nx, j=1,2,...,ny

    where a stands for all batch indices. y is a tensor with final shape n+1

    """

    def __init__(self):
        """Initialise new instance"""
        super().__init__()

    def forward(self, x):
        """Forward evaluation"""
        y = torch.empty(size=x.shape[:-2] + (x.shape[-2] + 1, x.shape[-1] + 1))
        # the four corners of the domain
        y[..., 0, 0] = x[..., 0, 0]
        y[..., -1, 0] = x[..., -1, 0]
        y[..., 0, -1] = x[..., 0, -1]
        y[..., -1, -1] = x[..., -1, -1]
        # central vertices
        y[..., 1:-1, 1:-1] = 0.25 * (
            x[..., :-1, :-1] + x[..., 1:, :-1] + x[..., :-1, 1:] + x[..., 1:, 1:]
        )
        # vertices on edge of domain
        y[..., 1:-1, 0] = 0.5 * (x[..., :-1, 0] + x[..., 1:, 0])
        y[..., 1:-1, -1] = 0.5 * (x[..., :-1, -1] + x[..., 1:, -1])
        y[..., 0, 1:-1] = 0.5 * (x[..., 0, :-1] + x[..., 0, 1:])
        y[..., -1, 1:-1] = 0.5 * (x[..., -1, :-1] + x[..., -1, 1:])
        return y
