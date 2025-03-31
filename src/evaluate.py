import itertools
import os
import sys
import tomllib
import torch
import time

import numpy as np
from matplotlib import pyplot as plt

from peds.diffusion_model_1d import DiffusionModel1d
from peds.diffusion_model_2d import DiffusionModel2d
from peds.distributions import (
    LogNormalDistribution1d,
    LogNormalDistribution2d,
    FibreDistribution2d,
)
from peds.quantity_of_interest import QoISampling1d, QoISampling2d
from peds.datasets import PEDSDataset, SavedDataset
from peds.peds_model import PEDSModel
from peds.interpolation_1d import (
    VertexToVolumeInterpolator1d,
    VolumeToVertexInterpolator1d,
)
from peds.interpolation_2d import (
    VertexToVolumeInterpolator2d,
    VolumeToVertexInterpolator2d,
)


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

dim = config["model"]["dimension"]
model_filename = config["model"]["filename"]
n = config["discretisation"]["n"]
scaling_factor = config["discretisation"]["scaling_factor"]
n_samples_train = config["data"]["n_samples_train"]
n_samples_valid = config["data"]["n_samples_valid"]
n_samples_test = config["data"]["n_samples_test"]
data_filename = config["data"]["filename"]
batch_size = config["train"]["batch_size"]
n_epoch = config["train"]["n_epoch"]
sample_points = config["qoi"]["sample_points"]
n_lowres = n // scaling_factor

if dim == 1:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    f_rhs = torch.ones(size=(n,), dtype=torch.float)
    if config["data"]["distribution"] == "log_normal":
        distribution = LogNormalDistribution1d(
            n, config["data"]["Lambda"], config["data"]["a_power"]
        )
    else:
        raise RuntimeError("Only log-normal distribution supported in 1d")
    DiffusionModel = DiffusionModel1d
    QoISampling = QoISampling1d
    VertexToVolumeInterpolator = VertexToVolumeInterpolator1d
    VolumeToVertexInterpolator = VolumeToVertexInterpolator1d
    AvgPool = torch.nn.AvgPool2d
    flatten_idx = -1
elif dim == 2:
    device = "cpu"
    f_rhs = torch.ones(size=(n, n), dtype=torch.float)
    if config["data"]["distribution"] == "log_normal":
        distribution = LogNormalDistribution2d(n, config["data"]["Lambda"])
    elif config["data"]["distribution"] == "fibre":
        distribution = FibreDistribution2d(
            n,
            config["data"]["n_fibres"],
            config["data"]["d_fibre_min"],
            config["data"]["d_fibre_max"],
            config["data"]["kdiff_background"],
            config["data"]["kdiff_fibre_min"],
            config["data"]["kdiff_fibre_max"],
        )
    else:
        raise RuntimeError(f"Unknown distribution: {config["data"]["distribution"]}")
    DiffusionModel = DiffusionModel2d
    QoISampling = QoISampling2d
    VertexToVolumeInterpolator = VertexToVolumeInterpolator2d
    VolumeToVertexInterpolator = VolumeToVertexInterpolator2d
    AvgPool = torch.nn.AvgPool2d
    flatten_idx = -2
else:
    raise RuntimeError(f"invalid dimension: {dim}")

print(f"Running on device {device}")

physics_model_highres = DiffusionModel(f_rhs)
qoi = QoISampling(sample_points)

n_samples = n_samples_train + n_samples_valid + n_samples_test
if not os.path.exists(data_filename):
    dataset = PEDSDataset(distribution, physics_model_highres, qoi)
    dataset.save(n_samples, data_filename)
dataset = SavedDataset(data_filename)
assert len(dataset) == n_samples

test_dataset = list(
    itertools.islice(
        dataset,
        n_samples_train + n_samples_valid,
        n_samples_train + n_samples_valid + n_samples_test,
    )
)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=n_samples_test)

physics_model_lowres = physics_model_highres.coarsen(scaling_factor)

downsampler = torch.nn.Sequential(
    torch.nn.Unflatten(flatten_idx, (1, n + 1)),
    VertexToVolumeInterpolator(),
    AvgPool(1, stride=scaling_factor),
    VolumeToVertexInterpolator(),
    torch.nn.Flatten(flatten_idx - 1, flatten_idx),
)


model = PEDSModel(physics_model_lowres, downsampler, qoi)
model.load(model_filename)

n_param = sum([torch.numel(p) for p in model.parameters()])
print(f"number of model parameters = {n_param}")

model = model.to(device)

sf = 2
coarse_model = dict()
while sf <= scaling_factor:
    coarse_model[sf] = torch.nn.Sequential(
        torch.nn.Sequential(
            torch.nn.Unflatten(flatten_idx, (1, n + 1)),
            VertexToVolumeInterpolator(),
            AvgPool(1, stride=sf),
            VolumeToVertexInterpolator(),
            torch.nn.Flatten(flatten_idx - 1, flatten_idx),
        ),
        physics_model_highres.coarsen(sf),
        qoi,
    )
    sf *= 2

loss_fn = torch.nn.MSELoss()

test_loss_avg = 0
for i, data in enumerate(test_dataloader):
    alpha, q_target = data
    alpha = alpha.to(device)
    q_target = q_target.to(device)
    q_pred = model(alpha)
    loss = loss_fn(q_pred, q_target)
    test_loss = loss.item()
    test_loss_avg += test_loss
    for sf, cm in coarse_model.items():
        q_pred_coarse = cm(alpha)
        coarse_loss = loss_fn(q_pred_coarse, q_target)
        print(f"coarse loss [{sf:d}x] = {coarse_loss:12.6f}")

data = next(iter(test_dataloader))
alpha, q_target = data
alpha = alpha.to(device)
q_target = q_target.to(device)
q_pred = model(alpha)

print(f"PEDS loss        = {test_loss_avg:12.6f}")

dataloader = torch.utils.data.DataLoader(dataset, batch_size=n_samples)
for data in test_dataloader:
    alpha, _ = data
    t_start = time.perf_counter()
    _ = model(alpha)
    t_finish = time.perf_counter()
    t_delta_peds = t_finish - t_start
    t_start = time.perf_counter()
    _ = physics_model_highres(alpha)
    t_finish = time.perf_counter()
    t_delta_petsc = t_finish - t_start
    for sf, cm in coarse_model.items():
        t_start = time.perf_counter()
        _ = cm(alpha)
        t_finish = time.perf_counter()
        t_delta_ = t_finish - t_start
        print(f"dt [coarse, {sf:d}] = {t_delta_:8.4f} s")

print(f"dt [PEDS]      = {t_delta_peds:8.4f} s")
print(f"dt [PETSc]     = {t_delta_petsc:8.4f} s")


# Visualise relative error

colors = ["blue", "red", "black", "green", "orange"]
sample_points = np.asanyarray(sample_points)
plt.clf()
if dim == 1:
    ax = plt.gca()
    plt.plot(
        sample_points,
        np.mean(abs(q_pred - q_target) / q_target, axis=0),
        linestyle="-",
        marker="o",
        markersize=6,
        label="PEDS",
    )
    plt.plot(
        sample_points,
        np.mean(abs(q_pred_coarse - q_target) / q_target, axis=0),
        marker="o",
        markersize=6,
        linestyle="--",
        label="coarse",
    )
    ax.set_yscale("log")
else:
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    error_coarse = np.mean(
        abs(q_pred_coarse.numpy() - q_target.numpy()) / q_target.numpy(), axis=0
    )
    error_peds = np.mean(
        abs(q_pred.detach().numpy() - q_target.numpy()) / q_target.numpy(), axis=0
    )
    ax.scatter(
        sample_points[:, 0],
        sample_points[:, 1],
        s=2000 * error_coarse,
        alpha=0.5,
        color="red",
    )
    ax.scatter(
        [],
        [],
        s=20,
        alpha=0.5,
        color="red",
        label="coarse",
    )
    for j in range(sample_points.shape[0]):
        ax.text(
            sample_points[j, 0] + 0.125 * error_coarse[j],
            sample_points[j, 1],
            f"{100*error_coarse[j]:6.1f}%",
            color="red",
            verticalalignment="top",
            size="small",
        )
        ax.text(
            sample_points[j, 0] + 0.125 * error_coarse[j],
            sample_points[j, 1],
            f"{100*error_peds[j]:6.1f}%",
            verticalalignment="bottom",
            color="blue",
            size="small",
        )
    ax.scatter(
        sample_points[:, 0],
        sample_points[:, 1],
        s=2000 * error_peds,
        alpha=0.5,
        color="blue",
    )
    ax.scatter(
        [],
        [],
        s=20,
        alpha=0.5,
        color="blue",
        label="PEDS",
    )


plt.legend(loc="upper left")
plt.savefig("evaluation.pdf", bbox_inches="tight")

# visualise solution
if dim == 1:
    coarse_u_model = torch.nn.Sequential(downsampler, physics_model_lowres)
    u_pred = model.get_u(alpha).cpu().detach().numpy()
    u_pred_coarse = coarse_u_model(alpha).cpu().detach().numpy()

    u_true = physics_model_highres(alpha.cpu()).detach().numpy()

    alpha = alpha.cpu().detach().numpy()

    X_alpha = np.arange(alpha.shape[-1]) / (alpha.shape[-1] - 1)
    X_u = (np.arange(u_pred.shape[-1]) + 0.5) / u_pred.shape[-1]
    X_u_true = (np.arange(u_true.shape[-1]) + 0.5) / u_true.shape[-1]

    for j in range(alpha.shape[0]):
        plt.clf()
        plt.plot(
            sample_points,
            q_target[j, :],
            linewidth=0,
            marker="o",
            markersize=6,
            color="green",
            label=r"$Q_{\text{true}}$",
        )
        plt.plot(
            X_alpha,
            0.1 * np.exp(alpha[j, :]),
            color="red",
            label=r"$\frac{1}{10}\exp[\alpha(x)]$",
        )
        plt.plot(
            X_u,
            u_pred[j, :],
            color="blue",
            linestyle="-",
            label=r"$u_{\text{PEDS}}(x)$",
        )
        plt.plot(
            X_u,
            u_pred_coarse[j, :],
            color="blue",
            linestyle="--",
            label=r"$u_{\text{coarse}}(x)$",
        )
        plt.plot(
            X_u_true,
            u_true[j, :],
            color="blue",
            linestyle=":",
            label=r"$u_{\text{highres}}(x)$",
        )
        plt.legend(loc="upper left")
        plt.savefig(f"solution_{j:03d}.pdf", bbox_inches="tight")
