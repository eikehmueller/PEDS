import torch


class Downsampler1d(torch.nn.Module):
    """Downsample a vector defined at grid points.

    Maps a vector of size n_in+1 to n_out+1 where n_in=factor*n_out must be an integer
    multiple of n_out
    """

    def __init__(self, scaling_factor):
        """Initialise new instance

        :arg scaling_factor: scaling factor
        """
        super().__init__()
        self.scaling_factor = scaling_factor

    def forward(self, x):
        return x[..., :: self.scaling_factor]
