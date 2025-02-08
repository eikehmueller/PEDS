import itertools
import torch
import pickle

__all__ = ["PEDSDataset","SavedDataset"]

class PEDSDataset(torch.utils.data.IterableDataset):
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
            alpha = torch.tensor(
                next(iter(self._distribution)), dtype=torch.torch.get_default_dtype()
            )
            u = self._physics_model(alpha)
            q = self._qoi(u)
            yield (alpha, q)

    def save(self,n_samples, filename):
        """Save dataset to disk
        
        :arg n_samples: number of samples to save
        :arg filename: name of file to save to
        """
        data = []
        for alpha, q in itertools.islice(iter(self),n_samples):
            data.append([alpha.numpy(),q.numpy()])
        with open(filename,"wb") as f:                        
            pickle.dump(data,f)
            

class SavedDataset(torch.utils.data.IterableDataset):
    """Dataset that can be read from disk"""
    def __init__(self,filename):
        with open(filename,"rb") as f:
            self._data = pickle.load(f)            

    def __iter__(self):
        """Return a new sample"""
        for alpha, q in self._data:            
            yield (torch.tensor(alpha), torch.tensor(q))
    
    def __len__(self):
        return len(self._data)
