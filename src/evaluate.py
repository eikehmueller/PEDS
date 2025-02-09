import itertools
import os
import tomllib
import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from matplotlib import pyplot as plt

from peds.diffusion_model import DiffusionModel1d
from peds.distributions import LogNormalDistribution1d
from peds.quantity_of_interest import QoISampling1d
from peds.datasets import PEDSDataset, SavedDataset
from peds.peds_model import PEDSModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on device {device}")

config_file = "config.toml"

with open(config_file, "rb") as f:
    config = tomllib.load(f)
print()
print(f"==== parameters ====")
print()
with open(config_file, "r") as f:
    for line in f.readlines():
        print(line.strip())
print()

use_peds = config["model"]["use_peds"] == "True"
model_filename = config["model"]["filename"]
n = config["discretisation"]["n"]
scaling_factor = config["discretisation"]["scaling_factor"]
Lambda = config["data"]["Lambda"]
a_power = config["data"]["a_power"]
n_samples_train = config["data"]["n_samples_train"]
n_samples_valid = config["data"]["n_samples_valid"]
n_samples_test = config["data"]["n_samples_test"]
data_filename = config["data"]["filename"]
batch_size = config["train"]["batch_size"]
n_epoch = config["train"]["n_epoch"]
sample_points = config["qoi"]["sample_points"]
n_lowres = n // scaling_factor

f_rhs = torch.ones(size=(n,), dtype=torch.float)

distribution = LogNormalDistribution1d(n, Lambda, a_power)
physics_model_highres = DiffusionModel1d(f_rhs)
qoi = QoISampling1d(sample_points)

n_samples = n_samples_train + n_samples_valid + n_samples_test
if not os.path.exists(data_filename):
    dataset = PEDSDataset(distribution, physics_model_highres, qoi)
    dataset.save(n_samples,data_filename)
dataset = SavedDataset(data_filename)


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
    torch.nn.Unflatten(-1, (1, n + 1)),
    torch.nn.AvgPool1d(1, stride=scaling_factor),
    torch.nn.Flatten(-2, -1),
)

nn_model = torch.load(model_filename, weights_only=False)
nn_model.eval()

if use_peds:
    model = PEDSModel(physics_model_lowres, nn_model, downsampler, qoi)
else:
    dense_layers = torch.nn.Sequential(
        torch.nn.Linear(n_lowres + 1, n_lowres),
        torch.nn.ReLU(),
        torch.nn.Linear(n_lowres, n_lowres),
        torch.nn.ReLU(),
        torch.nn.Linear(n_lowres, len(sample_points)),
    )
    model = nn_model + dense_layers
n_param = sum([torch.numel(p) for p in model.parameters()])
print(f"number of model parameters = {n_param}")

model = model.to(device)
coarse_model = torch.nn.Sequential(downsampler, physics_model_lowres, qoi)

loss_fn = torch.nn.MSELoss()

test_loss_avg = 0
for i, data in enumerate(test_dataloader):
    alpha, q_target = data
    alpha = alpha.to(device)
    q_target = q_target.to(device)
    q_pred = model(alpha)
    loss = loss_fn(q_pred, q_target)
    test_loss = loss.item()
    test_loss_avg += test_loss / (n_samples_test / batch_size)
    q_pred_coarse = coarse_model(alpha)
    coarse_loss = loss_fn(q_pred_coarse, q_target)

data = next(iter(test_dataloader))
alpha, q_target = data
alpha = alpha.to(device)
q_target = q_target.to(device)
q_pred = model(alpha)

q_pred_coarse = coarse_model(alpha)
q_pred = q_pred.cpu().detach().numpy()
q_pred_coarse = q_pred_coarse.cpu().detach().numpy()
q_target = q_target.cpu().detach().numpy()

print(f"test loss = {test_loss_avg:12.6f} coarse loss = {coarse_loss:12.6f}")

# Visualise relative error

colors = ["blue", "red", "black", "green", "orange"]
plt.clf()
plt.plot(sample_points,
        np.mean(abs(q_pred - q_target) / q_target,axis=0), linestyle="-", marker="o",markersize=6,label="PEDS",
    )
plt.plot(sample_points,
        np.mean(abs(q_pred_coarse - q_target) / q_target,axis=0),marker="o",markersize=6,
        linestyle="--",label="coarse"
    )    
ax = plt.gca()
#ax.set_yscale("log")
plt.legend(loc="upper right")
plt.savefig("evaluation.pdf", bbox_inches="tight")

# visualise solution

coarse_u_model = torch.nn.Sequential(downsampler, physics_model_lowres)
u_pred = model.get_u(alpha).cpu().detach().numpy()
u_pred_coarse = coarse_u_model(alpha).cpu().detach().numpy()
alpha = alpha.cpu().detach().numpy()
X_alpha = np.arange(alpha.shape[-1])/(alpha.shape[-1]-1)
X_u = (np.arange(u_pred.shape[-1])+0.5)/u_pred.shape[-1]

for j in range(alpha.shape[0]):
    plt.clf()
    plt.plot(sample_points,q_target[j,:],linewidth=0,marker="o",markersize=6,color="green",label=r"$Q_{\text{true}}$")
    plt.plot(X_alpha,0.1*np.exp(alpha[j,:]),color="red",label=r"$\frac{1}{10}\exp[\alpha(x)]$")
    plt.plot(X_u,u_pred[j,:],color="blue",linestyle="-",label=r"$u_{\text{PEDS}}(x)$")
    plt.plot(X_u,u_pred_coarse[j,:],color="blue",linestyle="--",label=r"$u_{\text{coarse}}(x)$")
    plt.legend(loc="upper left")
    plt.savefig(f"solution_{j:03d}.pdf", bbox_inches="tight")

