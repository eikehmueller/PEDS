"""Script for evaluating and visualising the performance of the PEDS model"""

import time
import copy
import torch

import numpy as np
from matplotlib import pyplot as plt

from peds.peds_model import PEDSModel

from setup import (
    read_config,
    get_distribution,
    get_physics_model,
    get_qoi,
    get_downsampler,
    get_coarse_model,
    get_datasets,
)


def measure_error(
    dataset, peds_model, physics_model_highres, scaling_factor, sum_qoi_components=False
):
    """Measure loss values of different models

    Measure the loss function for the PEDS model, the high-fidelity physics model
    and coarsened versions of the physics model.

    Returns a dictionary with the error values for each model.

    :arg dataset: dataset to use for the loss calculation
    :arg peds_model: PEDS model
    :arg physics_model_highres: high-fidelity physics model
    :arg scaling_factor: scaling factor for the coarsest model
    :arg sum_qoi_components: if True, sum the components of the error
    """
    sfs = 2 ** (1 + np.arange(int(np.log2(scaling_factor))))
    coarse_model = {sf: get_coarse_model(physics_model_highres, sf, qoi) for sf in sfs}
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    rms_error = dict()
    alpha, q_target = next(iter(dataloader))
    q_pred = peds_model(alpha)
    avg_squared_error = np.average((q_pred - q_target).detach().numpy() ** 2, axis=0)
    if sum_qoi_components:
        rms_error["peds"] = np.sqrt(np.sum(avg_squared_error))
    else:
        rms_error["peds"] = np.sqrt(avg_squared_error)
    rms_error["coarse"] = dict()
    for sf, cm in coarse_model.items():
        q_pred_coarse = cm(alpha)
        avg_squared_error = np.average(
            (q_pred_coarse - q_target).detach().numpy() ** 2, axis=0
        )
        if sum_qoi_components:
            rms_error["coarse"][sf] = np.sqrt(np.sum(avg_squared_error))
        else:
            rms_error["coarse"][sf] = np.sqrt(avg_squared_error)
    return rms_error


def measure_performance(
    dataset, peds_model, physics_model_highres, scaling_factor, device
):
    """Measure the performance of the models

    Measure the time it takes to compute a single forward pass for the PEDS model, the
    high-fidelity physics model and coarsened versions of the physics model.

    Returns a dictonary with the time-per-sample values for each model.

    :arg dataset: dataset to use for the loss calculation
    :arg peds_model: PEDS model
    :arg physics_model_highres: high-fidelity physics model
    :arg scaling_factor: scaling factor for the coarsest model
    :arg device: device to run on
    """
    if device == next(peds_model.parameters()).device:
        physics_model_highres_device = physics_model_highres
        peds_model_device = peds_model
    else:
        physics_model_highres_device = copy.deepcopy(physics_model_highres)
        physics_model_highres_device.to(device)
        peds_model_device = copy.deepcopy(peds_model)
        peds_model_device.to(device)
    n_samples = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=n_samples)
    sfs = 2 ** (1 + np.arange(int(np.log2(scaling_factor))))
    coarse_model_device = {
        sf: get_coarse_model(physics_model_highres_device, sf, qoi).to(device)
        for sf in sfs
    }
    time_per_sample = dict()
    alpha, _ = next(iter(dataloader))

    alpha = alpha.to(device)
    t_start = time.perf_counter()
    _ = peds_model_device(alpha)
    t_finish = time.perf_counter()
    time_per_sample["peds"] = (t_finish - t_start) / n_samples
    t_start = time.perf_counter()
    _ = physics_model_highres_device(alpha)
    t_finish = time.perf_counter()
    time_per_sample["fine"] = (t_finish - t_start) / n_samples
    time_per_sample["coarse"] = dict()
    for sf, cm in coarse_model_device.items():
        t_start = time.perf_counter()
        _ = cm(alpha)
        t_finish = time.perf_counter()
        time_per_sample["coarse"][sf] = (t_finish - t_start) / n_samples
    return time_per_sample


def visualise_performance(rms_error, time_per_sample, filename):
    """Plot the error per sample vs. time per sample for different models

    :rms_error: dictionary with the RMSE error for each model,
        computed by measure_error
    :time_per_sample: dictionary with the time per sample for each model,
        computed by measure_performance
    :filename: name of file to save the plot to
    """
    plt.clf()

    plt.plot(
        [time_per_sample["peds"]],
        [rms_error["peds"]],
        color="red",
        linewidth=2,
        marker="o",
        markersize=6,
        label="PEDS",
    )
    sfs = sorted(rms_error["coarse"].keys())
    plt.plot(
        [time_per_sample["coarse"][sf] for sf in sfs],
        [rms_error["coarse"][sf] for sf in sfs],
        linewidth=2,
        color="blue",
        marker="o",
        markersize=6,
        label="coarse",
    )
    plt.plot(
        [time_per_sample["fine"], time_per_sample["coarse"][2]],
        [0, rms_error["coarse"][2]],
        color="blue",
        linewidth=2,
        linestyle="--",
    )
    plt.plot(
        [time_per_sample["fine"]],
        [0],
        color="blue",
        marker="o",
        linewidth=2,
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=2,
        label="highres",
    )

    for sf in sfs:
        plt.annotate(
            f"   ${sf}\\times $",
            xy=(time_per_sample["coarse"][sf], rms_error["coarse"][sf]),
        )

    ax = plt.gca()
    ax.set_xlabel("time per sample [ms]")
    ax.set_ylabel("RMS error")
    ax.set_xscale("log")
    plt.legend(loc="upper right")
    plt.savefig(filename, bbox_inches="tight")


def visualise_error(rms_error, sample_points, filename):
    """Visualise the RMSE error

    :rms_error: dictionary with the RMSE error for each model
    :dataset: dataset to use for the error calculation
    :sample_points: Sample points used for for QoI
    :filename: name of file to save the plot to
    """
    sample_points = np.asanyarray(sample_points)
    scaling_factor = int(np.max(list(rms_error["coarse"].keys())))
    dim = sample_points.ndim
    plt.clf()
    ax = plt.gca()
    if dim == 1:
        plt.plot(
            sample_points,
            rms_error["peds"],
            linewidth=2,
            marker="o",
            markersize=6,
            label="PEDS",
        )
        plt.plot(
            sample_points,
            rms_error["coarse"][scaling_factor],
            marker="o",
            markersize=6,
            linewidth=2,
            label="coarse",
        )
        ax.set_yscale("log")
        ax.set_xlabel("sample point coordinate $x$")
        ax.set_ylabel("RMS error")
    else:
        _, ax = plt.subplots()
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.scatter(
            sample_points[:, 0],
            sample_points[:, 1],
            s=2000 * rms_error["coarse"][scaling_factor],
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
                sample_points[j, 0] + 0.125 * rms_error["coarse"][scaling_factor][j],
                sample_points[j, 1],
                f"{100*rms_error["coarse"][scaling_factor][j]:6.1f}%",
                color="red",
                verticalalignment="top",
                size="small",
            )
            ax.text(
                sample_points[j, 0] + 0.125 * rms_error["coarse"][scaling_factor][j],
                sample_points[j, 1],
                f"{100*rms_error["peds"][j]:6.1f}%",
                verticalalignment="bottom",
                color="blue",
                size="small",
            )
        ax.scatter(
            sample_points[:, 0],
            sample_points[:, 1],
            s=2000 * rms_error["peds"],
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
    plt.savefig(filename, bbox_inches="tight")


#####################################################################
#########################      M A I N      #########################
#####################################################################

if __name__ == "__main__":
    config = read_config()

    device = torch.device(
        "cuda:0"
        if config["model"]["dimension"] == 1 and torch.cuda.is_available()
        else "cpu"
    )
    # device = "cpu"

    print(f"Running on device {device}")

    distribution = get_distribution(config)
    physics_model_highres = get_physics_model(config)
    qoi = get_qoi(config)
    downsampler = get_downsampler(config)

    _, __, test_dataset = get_datasets(config)

    n_samples_test = config["data"]["n_samples_test"]
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=n_samples_test
    )

    scaling_factor = config["discretisation"]["scaling_factor"]
    physics_model_lowres = physics_model_highres.coarsen(scaling_factor)

    model = PEDSModel(physics_model_lowres, downsampler, qoi)
    model.load(config["model"]["filename"])

    n_param = sum([torch.numel(p) for p in model.parameters()])
    print(f"number of model parameters = {n_param}")

    rms_error = measure_error(
        test_dataset,
        model,
        physics_model_highres,
        scaling_factor,
        sum_qoi_components=True,
    )

    print()
    print("==== error ====")
    for key, rmse in rms_error.items():
        if key == "coarse":
            for scaling_factor, coarse_rmse in rmse.items():
                print(
                    f"  rmse error [coarse {scaling_factor:2d}x] = {coarse_rmse:8.4e}"
                )
        else:
            print(f"  rmse error [{key:10s}] = {rmse:8.4e}")

    time_per_sample = measure_performance(
        test_dataset, model, physics_model_highres, scaling_factor, device
    )
    print()
    print("==== performance ====")
    for key, value in time_per_sample.items():
        if key == "coarse":
            for scaling_factor, value in value.items():
                print(
                    f"  time per sample [coarse {scaling_factor:2d}x] = {1000*value:8.4e} ms"
                )
        else:
            print(f"  time per sample [{key:10s}] = {1000*value:8.4e} ms")
    print()

    visualise_performance(rms_error, time_per_sample, "performance.pdf")

    rms_error = measure_error(
        test_dataset,
        model,
        physics_model_highres,
        scaling_factor,
        sum_qoi_components=False,
    )

    visualise_error(rms_error, config["qoi"]["sample_points"], "rmse.pdf")
