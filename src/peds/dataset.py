import torch


class Dataset(torch.utils.data.IterableDataset):
    """Dataset consisting of (geometry, QoI) pairs"""

    def __init__(self, distribution, physics_model, qoi):
        """Initialise new instance

        :arg distribution: distribution from which the geometry is drawn
        :arg physics_model: model that maps geometry to a solution
        :arg qoi: quantity of interest
        """
        super().__init__()
        self._distribution = distribution
        self._physics_model = physics_model
        self._qoi = qoi

    def __iter__(self):
        """Return a new sample"""
        while True:
            alpha = torch.tensor(next(iter(self._distribution)))
            u = self._physics_model(alpha)
            q = self._qoi(u)
            yield (alpha, q)
