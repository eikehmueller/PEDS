import itertools
import os
import torch
import time

import numpy as np
from matplotlib import pyplot as plt

from peds.datasets import PEDSDataset, SavedDataset
from peds.peds_model import PEDSModel

from common import (
    read_config
    get_distribution,
    get_physics_model,
    get_qoi,
    get_downsampler,
    get_coarse_model,
)

config = read_config()


device = torch.device(
    "cuda:0" if config["model"]["dimension"] == 1and torch.cuda.is_available() else "cpu"
)

print(f"Running on device {device}")

distribution = get_distribution(config)
physics_model_highres = get_physics_model(config)
qoi = get_qoi(config)
downsampler = get_downsampler(config)


n_samples_train = config["data"]["n_samples_train"]
n_samples_valid = config["data"]["n_samples_valid"]
n_samples_test = config["data"]["n_samples_test"]
n_samples = n_samples_train + n_samples_valid + n_samples_test

data_filename = config["data"]["filename"]
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

physics_model_lowres = physics_model_highres.coarsen(
    config["discretisation"]["scaling_factor"]
)


model = PEDSModel(physics_model_lowres, downsampler, qoi)
model.load(config["model"]["filename"])

n_param = sum([torch.numel(p) for p in model.parameters()])
print(f"number of model parameters = {n_param}")

model = model.to(device)

scaling_factor = 2
coarse_model = dict()
while scaling_factor <= config["discretisation"]["scaling_factor"]:
    coarse_model[scaling_factor] = get_coarse_model(
        physics_model_highres, scaling_factor, qoi
    )
    scaling_factor *= 2


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
    coarse_loss = dict()
    for scaling_factor, cm in coarse_model.items():
        q_pred_coarse = cm(alpha)
        coarse_loss[scaling_factor] = loss_fn(q_pred_coarse, q_target).detach().item()
        print(
            f"coarse loss [{scaling_factor:d}x] = {coarse_loss[scaling_factor]:12.6f}"
        )

data = next(iter(test_dataloader))
alpha, q_target = data
alpha = alpha.to(device)
q_target = q_target.to(device)
q_pred = model(alpha)

print(f"PEDS loss        = {test_loss_avg:12.6f}")

dataloader = torch.utils.data.DataLoader(dataset, batch_size=n_samples_test)
for data in test_dataloader:
    alpha, _ = data
    t_start = time.perf_counter()
    _ = model(alpha)
    t_finish = time.perf_counter()
    t_delta_peds = 1000 * (t_finish - t_start) / n_samples_test
    t_start = time.perf_counter()
    _ = physics_model_highres(alpha)
    t_finish = time.perf_counter()
    t_delta_fine = 1000 * (t_finish - t_start) / n_samples_test
    t_delta_coarse = dict()
    for scaling_factor, cm in coarse_model.items():
        t_start = time.perf_counter()
        _ = cm(alpha)
        t_finish = time.perf_counter()
        t_delta_coarse[scaling_factor] = 1000 * (t_finish - t_start) / n_samples_test
        print(
            f"dt [coarse, {scaling_factor:d}] = {t_delta_coarse[scaling_factor]:8.4f} ms"
        )

print(f"dt [PEDS]      = {t_delta_peds:8.4f} ms")
print(f"dt [fine]     = {t_delta_fine:8.4f} ms")

plt.clf()

plt.plot(
    [t_delta_peds],
    [np.sqrt(test_loss_avg)],
    color="red",
    linewidth=2,
    marker="o",
    markersize=6,
    label="PEDS",
)
plt.plot(
    [t_delta_coarse[sf] for sf in sorted(t_delta_coarse.keys())],
    [np.sqrt(coarse_loss[sf]) for sf in sorted(coarse_loss.keys())],
    linewidth=2,
    color="blue",
    marker="o",
    markersize=6,
    label="coarse",
)
plt.plot(
    [t_delta_fine, t_delta_coarse[2]],
    [0, np.sqrt(coarse_loss[2])],
    color="blue",
    linewidth=2,
    linestyle="--",
)
plt.plot(
    [t_delta_fine],
    [0],
    color="blue",
    marker="o",
    linewidth=2,
    markersize=6,
    markerfacecolor="white",
    markeredgewidth=2,
    label="highres",
)

for scaling_factor in sorted(t_delta_coarse.keys()):
    plt.annotate(
        f"   ${scaling_factor}\\times $",
        xy=(t_delta_coarse[scaling_factor], np.sqrt(coarse_loss[scaling_factor])),
    )


ax = plt.gca()
ax.set_xlabel("time per sample [ms]")
ax.set_ylabel("error")
ax.set_xscale("log")
plt.legend(loc="upper right")
plt.savefig("performance.pdf", bbox_inches="tight")

# Visualise relative error

colors = ["blue", "red", "black", "green", "orange"]
sample_points = np.asanyarray(sample_points)
plt.clf()
if dim == 1:
    ax = plt.gca()
    plt.plot(
        sample_points,
        np.mean(abs(q_pred - q_target) / q_target, axis=0),
        linewidth=2,
        marker="o",
        markersize=6,
        label="PEDS",
    )
    plt.plot(
        sample_points,
        np.mean(abs(q_pred_coarse - q_target) / q_target, axis=0),
        marker="o",
        markersize=6,
        linewidth=2,
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
