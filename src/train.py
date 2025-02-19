import itertools
import os.path
import tomllib
import torch
from torch.utils.tensorboard import SummaryWriter

from peds.diffusion_model import DiffusionModel1d
from peds.distributions import LogNormalDistribution1d
from peds.quantity_of_interest import QoISampling1d
from peds.datasets import PEDSDataset, SavedDataset
from peds.peds_model import PEDSModel
from peds.interpolation_1d import (
    VertexToVolumeInterpolator1d,
    VolumeToVertexInterpolator1d,
)

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
lr_initial = config["train"]["lr_initial"]
lr_target = config["train"]["lr_final"]
sample_points = config["qoi"]["sample_points"]
n_lowres = n // scaling_factor

f_rhs = torch.ones(size=(n,), dtype=torch.float)

distribution = LogNormalDistribution1d(n, Lambda, a_power)
physics_model_highres = DiffusionModel1d(f_rhs)
qoi = QoISampling1d(sample_points)
n_samples = n_samples_train + n_samples_valid + n_samples_test
if not os.path.exists(data_filename):
    dataset = PEDSDataset(distribution, physics_model_highres, qoi)
    dataset.save(n_samples, data_filename)
dataset = SavedDataset(data_filename)
assert len(dataset) == n_samples

train_dataset = list(itertools.islice(dataset, 0, n_samples_train))
valid_dataset = list(
    itertools.islice(dataset, n_samples_train, n_samples_train + n_samples_valid)
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
valid_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=n_samples_valid
)


physics_model_lowres = physics_model_highres.coarsen(scaling_factor)

downsampler = torch.nn.Sequential(
    torch.nn.Unflatten(-1, (1, n + 1)),
    VertexToVolumeInterpolator1d(),
    torch.nn.AvgPool1d(1, stride=scaling_factor),
    VolumeToVertexInterpolator1d(),
    torch.nn.Flatten(-2, -1),
)

nn_model = torch.nn.Sequential(
    torch.nn.Unflatten(-1, (1, n + 1)),
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
    torch.nn.Conv1d(8, 1, 3, padding=1),
    torch.nn.Flatten(-2, -1),
)

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
gamma = (lr_target / lr_initial) ** (1 / n_epoch)
print(f"learning rate decay factor = {gamma:8.5f}")
optimizer = torch.optim.Adam(model.parameters(), lr=lr_initial)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

writer = SummaryWriter(flush_secs=5)
print(f"epoch      :  training    validation    coarse")
for epoch in range(n_epoch):
    train_loss_avg = 0
    for i, data in enumerate(train_dataloader):
        alpha, q_target = data
        alpha = alpha.to(device)
        q_target = q_target.to(device)
        optimizer.zero_grad()
        q_pred = model(alpha)
        loss = loss_fn(q_pred, q_target)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        train_loss_avg += train_loss / (n_samples_train / batch_size)
    alpha, q_target = next(iter(valid_dataloader))
    alpha = alpha.to(device)
    q_target = q_target.to(device)
    q_pred = model(alpha)
    valid_loss = loss_fn(q_pred, q_target)
    q_pred_coarse = coarse_model(alpha)
    coarse_loss = loss_fn(q_pred_coarse, q_target)
    writer.add_scalars(
        "Loss",
        {"train": train_loss_avg, "valid": valid_loss, "coarse": coarse_loss},
        epoch,
    )
    if isinstance(model, PEDSModel):
        writer.add_scalar("NN weight", model.w.detach(), epoch)
    writer.add_scalar("gain", coarse_loss / valid_loss, epoch)
    writer.add_scalar("learning rate", scheduler.get_last_lr()[0], epoch)
    print(
        f"epoch {epoch+1:5d}:  {train_loss_avg:12.6f} {valid_loss:12.6f} {coarse_loss:12.6f}"
    )
    scheduler.step()

writer.flush()

torch.save(nn_model, model_filename)
