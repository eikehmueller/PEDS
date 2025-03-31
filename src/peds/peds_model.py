import os
import json
import torch


class PEDSModel(torch.nn.Module):
    """Physics enhanced deep surrogate model

    Reference:
        Pestourie, R., Mroueh, Y., Rackauckas, C., Das, P. and Johnson, S.G., 2023.
        "Physics-enhanced deep surrogates for partial differential equations."
        Nature Machine Intelligence, 5(12), pp.1458-1465.
    """

    def __init__(self, physics_model, downsampler, qoi, nn_model=None):
        """Initialise new instance

        :arg physics_model: physical model which maps the geometry to a solution
        :arg downsampler: downsampler from high-res geometry to low-res
            geometry
        :arg qoi: quantity of interest
        :arg nn_model: neural network model that maps the high-res geometry
            to the low-res geometry
        """
        super().__init__()
        self._physics_model = physics_model
        self._nn_model = nn_model
        self._downsampler = downsampler
        self._qoi = qoi
        self.w = torch.nn.Parameter(torch.Tensor([0.5]))
        self.w.requires_grad = True

    def to(self, device):
        """Move to device

        :arg device: device to move to
        """
        new_self = super().to(device)
        new_self._physics_model = self._physics_model.to(device)
        new_self._downsampler = self._downsampler.to(device)
        new_self._nn_model = self._nn_model.to(device)
        new_self.w = self.w.to(device)
        return new_self

    def forward(self, x):
        """Evaluate model"""
        u = self.get_u(x)
        return self._qoi(u)

    def get_u(self, x):
        """Evaluate model"""
        alpha_ds = self._downsampler(x)
        alpha_nn = self._nn_model(x)
        alpha = self.w * alpha_nn + (1 - self.w) * alpha_ds
        return self._physics_model(alpha)

    def save(self, filename):
        """Save nn model and weight to disk

        :arg filename: name of file to save to
        """
        torch.save(self._nn_model, filename)
        with open(
            os.path.splitext(filename)[0] + "_nnweight.json", "w", encoding="utf8"
        ) as f:
            json.dump(dict(weight=self.w.detach().item()), f)

    def load(self, filename):
        """load nn model and weight from disk

        :arg filename: name of file to load from to
        """
        self._nn_model = torch.load(filename, weights_only=False)
        self._nn_model.eval()

        with open(
            os.path.splitext(filename)[0] + "_nnweight.json", encoding="utf8"
        ) as f:
            data = json.load(f)
            self.w = torch.nn.Parameter(torch.Tensor([data["weight"]]))
            self.w.requires_grad = True
