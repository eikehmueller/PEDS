"""Common setup routines for training and evaluation"""

import sys
import tomllib
import itertools
import torch

from peds.datasets import SavedDataset
from peds.diffusion_model_1d import DiffusionModel1d
from peds.diffusion_model_2d import DiffusionModel2d
from peds.distributions_lognormal import (
    LogNormalDistribution1d,
    LogNormalDistribution2d,
)
from peds.distributions_fibres import FibreRadiusDistribution, FibreDistribution2d
from peds.quantity_of_interest import QoISampling1d, QoISampling2d
from peds.interpolation_1d import (
    VertexToVolumeInterpolator1d,
    VolumeToVertexInterpolator1d,
)
from peds.interpolation_2d import (
    VertexToVolumeInterpolator2d,
    VolumeToVertexInterpolator2d,
)

__all__ = [
    "read_config",
    "get_distribution",
    "get_physics_model",
    "get_qoi",
    "get_downsampler",
    "get_nn_model",
    "get_coarse_model",
    "get_pure_nn_model",
    "get_datasets",
]


def read_config():
    """Parse command line arguments and read configuration file"""
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} CONFIGFILE")
        sys.exit(0)

    config_file = sys.argv[1]
    print(f"reading parameters from {config_file}")

    with open(config_file, "rb") as f:
        config = tomllib.load(f)
    print()
    print("==== parameters ====")
    print()
    with open(config_file, "r", encoding="utf8") as f:
        for line in f.readlines():
            print(line.strip())
    print()
    return config


def get_distribution(config):
    """Initialise the distribution based on configuration

    Returns distribution object that can be used to generate samples for the logarithm of
    the diffusion coefficient.

    :arg config: configuration dictionary
    """
    n = config["discretisation"]["n"]
    domain_size = config["discretisation"]["domain_size"]
    dim = config["model"]["dimension"]
    if dim == 1:
        if config["data"]["distribution"] == "lognormal":
            return LogNormalDistribution1d(
                n,
                domain_size,
                config["distribution"]["lognormal"]["Lambda"],
                config["distribution"]["lognormal"]["a_power"],
            )
        else:
            raise RuntimeError("Only log-normal distribution supported in 1d")
    elif dim == 2:
        if config["data"]["distribution"] == "lognormal":
            return LogNormalDistribution2d(
                n, domain_size, config["distribution"]["lognormal"]["Lambda"]
            )
        elif config["data"]["distribution"] == "fibre":
            r_fibre_dist = FibreRadiusDistribution(
                r_avg=config["distribution"]["fibre"]["r_fibre_avg"],
                r_min=config["distribution"]["fibre"]["r_fibre_min"],
                r_max=config["distribution"]["fibre"]["r_fibre_max"],
                sigma=config["distribution"]["fibre"]["r_fibre_sigma"],
                gaussian=config["distribution"]["fibre"]["gaussian"],
            )
            return FibreDistribution2d(
                n,
                domain_size,
                volume_fraction=config["distribution"]["fibre"]["volume_fraction"],
                r_fibre_dist=r_fibre_dist,
                kdiff_background=config["distribution"]["fibre"]["kdiff_background"],
                kdiff_fibre=config["distribution"]["fibre"]["kdiff_fibre"],
            )
    else:
        raise RuntimeError(f"invalid dimension: {dim}")


def get_physics_model(config):
    """Construct the high-resolution physics model based on the configuration

    Returns physics model

    :arg config: configuration dictionary
    """
    n = config["discretisation"]["n"]
    domain_size = config["discretisation"]["domain_size"]
    dim = config["model"]["dimension"]
    if dim == 1:
        f_rhs = torch.ones(size=(n,), dtype=torch.float)
        return DiffusionModel1d(f_rhs, domain_size)
    elif dim == 2:
        f_rhs = torch.ones(size=(n, n), dtype=torch.float)
        return DiffusionModel2d(f_rhs, domain_size)
    else:
        raise RuntimeError(f"invalid dimension: {dim}")


def get_qoi(config):
    """Initialise the QoI based on the configuration

    Returns QoI object that can be used to sample the quantity of interest
    """
    dim = config["model"]["dimension"]
    if dim == 1:
        return QoISampling1d(config["qoi"]["sample_points"])
    elif dim == 2:
        return QoISampling2d(config["qoi"]["sample_points"])
    else:
        raise RuntimeError(f"invalid dimension: {dim}")


def get_downsampler(config):
    """Initialise the downsampler based on the configuration

    Returns downsampler object that can be used to downsample the high-resolution field

    :arg config: configuration dictionary
    """
    n = config["discretisation"]["n"]
    dim = config["model"]["dimension"]
    if dim == 1:
        return torch.nn.Sequential(
            torch.nn.Unflatten(-1, (1, n + 1)),
            VertexToVolumeInterpolator1d(),
            torch.nn.AvgPool1d(1, stride=config["discretisation"]["scaling_factor"]),
            VolumeToVertexInterpolator1d(),
            torch.nn.Flatten(-2, -1),
        )

    elif dim == 2:
        return torch.nn.Sequential(
            torch.nn.Unflatten(-2, (1, n + 1)),
            VertexToVolumeInterpolator2d(),
            torch.nn.AvgPool2d(1, stride=config["discretisation"]["scaling_factor"]),
            VolumeToVertexInterpolator2d(),
            torch.nn.Flatten(-3, -2),
        )
    else:
        raise RuntimeError(f"invalid dimension: {dim}")


def get_nn_model(config):
    """Initialise the NN model based on the configuration

    Returns neural network model object

    :arg config: configuration dictionary
    """
    n = config["discretisation"]["n"]
    if config["model"]["dimension"] == 1:
        return torch.nn.Sequential(
            torch.nn.Unflatten(-1, (1, n + 1)),
            VertexToVolumeInterpolator1d(),
            torch.nn.Conv1d(1, 4, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, ceil_mode=True),
            torch.nn.Conv1d(4, 4, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, ceil_mode=True),
            torch.nn.Conv1d(4, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, ceil_mode=True),
            torch.nn.Conv1d(8, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, ceil_mode=True),
            torch.nn.Conv1d(8, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(8, 1, 3, padding=1),
            VolumeToVertexInterpolator1d(),
            torch.nn.Flatten(-2, -1),
        )

    elif config["model"]["dimension"] == 2:
        return torch.nn.Sequential(
            torch.nn.Unflatten(-2, (1, n + 1)),
            VertexToVolumeInterpolator2d(),
            torch.nn.Conv2d(1, 4, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, ceil_mode=True),
            torch.nn.Conv2d(4, 4, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, ceil_mode=True),
            torch.nn.Conv2d(4, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, ceil_mode=True),
            torch.nn.Conv2d(8, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, ceil_mode=True),
            torch.nn.Conv2d(8, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, 3, padding=1),
            VolumeToVertexInterpolator2d(),
            torch.nn.Flatten(-3, -2),
        )
    else:
        dim = config["model"]["dimension"]
        raise RuntimeError(f"invalid dimension: {dim}")


def get_coarse_model(physics_model_highres, scaling_factor, qoi):
    """Construct coarsened model based on scaling factor

    This model coarsens the diffusion coefficient and then applies a low-resolution
    physics model on which the quantity of interest is evaluated

    :arg physics_model_highres: high-resolution physics model
    :arg scaling_factor: scaling factor for the coarsening
    :arg qoi: quantity of interest object
    """
    f_rhs = physics_model_highres.metadata["f_rhs"]
    n = f_rhs.shape[-1]
    dim = f_rhs.ndim
    if dim == 1:
        return torch.nn.Sequential(
            torch.nn.Unflatten(-1, (1, n + 1)),
            VertexToVolumeInterpolator1d(),
            torch.nn.AvgPool1d(1, stride=scaling_factor),
            VolumeToVertexInterpolator1d(),
            torch.nn.Flatten(-2, -1),
            physics_model_highres.coarsen(scaling_factor),
            qoi,
        )
    elif dim == 2:
        return torch.nn.Sequential(
            torch.nn.Unflatten(-2, (1, n + 1)),
            VertexToVolumeInterpolator2d(),
            torch.nn.AvgPool2d(1, stride=scaling_factor),
            VolumeToVertexInterpolator2d(),
            torch.nn.Flatten(-3, -2),
            physics_model_highres.coarsen(scaling_factor),
            qoi,
        )
    else:
        raise RuntimeError(f"invalid dimension: {dim}")

def get_pure_nn_model(config):
    """Initialise the naive NN model

    Returns neural network model object

    :arg config: configuration dictionary
    """
    qoi = get_qoi(config)
    n = config["discretisation"]["n"]
    if config["model"]["dimension"] == 1:
        return torch.nn.Sequential(
            torch.nn.Unflatten(-1, (1, n + 1)),
            VertexToVolumeInterpolator1d(),
            torch.nn.Conv1d(1, 4, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, ceil_mode=True),
            torch.nn.Conv1d(4, 4, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, ceil_mode=True),
            torch.nn.Conv1d(4, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, ceil_mode=True),
            torch.nn.Conv1d(8, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, ceil_mode=True),
            torch.nn.Conv1d(8, 8, 3, padding=1),
            torch.nn.Flatten(-2, -1),
            torch.nn.ReLU(),
            torch.nn.Linear(n//2,16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,qoi.dim),
        )

    elif config["model"]["dimension"] == 2:
        return torch.nn.Sequential(
            torch.nn.Unflatten(-2, (1, n + 1)),
            VertexToVolumeInterpolator2d(),
            torch.nn.Conv2d(1, 4, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, ceil_mode=True),
            torch.nn.Conv2d(4, 4, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, ceil_mode=True),
            torch.nn.Conv2d(4, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, ceil_mode=True),
            torch.nn.Conv2d(8, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, ceil_mode=True),
            torch.nn.Conv2d(8, 8, 3, padding=1),
            torch.nn.Flatten(-3, -1),
            torch.nn.ReLU(),
            torch.nn.Linear(8*(n//16)**2,16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,qoi.dim),
        )
    else:
        dim = config["model"]["dimension"]
        raise RuntimeError(f"invalid dimension: {dim}")

def get_datasets(config):
    """Return the training, validation and test datasets

    The datasets are created from the saved dataset file.

    :arg config: configuration dictionary
    """
    n_samples_train = config["data"]["n_samples_train"]
    n_samples_valid = config["data"]["n_samples_valid"]
    n_samples_test = config["data"]["n_samples_test"]
    n_samples = n_samples_train + n_samples_valid + n_samples_test

    data_filename = config["data"]["filename"]
    dataset = SavedDataset(data_filename)
    assert len(dataset) == n_samples

    train_dataset = list(itertools.islice(dataset, 0, n_samples_train))
    valid_dataset = list(
        itertools.islice(dataset, n_samples_train, n_samples_train + n_samples_valid)
    )
    test_dataset = list(
        itertools.islice(
            dataset,
            n_samples_train + n_samples_valid,
            n_samples_train + n_samples_valid + n_samples_test,
        )
    )
    return train_dataset, valid_dataset, test_dataset
