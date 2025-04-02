import itertools
import os.path
import sys
import tomllib
import torch
from torch.utils.tensorboard import SummaryWriter

from peds.datasets import PEDSDataset, SavedDataset
from peds.peds_model import PEDSModel
from common import (
    get_distribution,
    get_physics_model,
    get_qoi,
    get_downsampler,
    get_nn_model,
)

if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} CONFIGFILE")
    sys.exit(0)

config_file = sys.argv[1]
print(f"reading parameters from {config_file}")

with open(config_file, "rb") as f:
    config = tomllib.load(f)
print()
print(f"==== parameters ====")
print()
with open(config_file, "r") as f:
    for line in f.readlines():
        print(line.strip())
print()


device = torch.device(
    "cuda:0" if config["model"]["dimension"] and torch.cuda.is_available() else "cpu"
)

print(f"Running on device {device}")

distribution = get_distribution(config)
physics_model_highres = get_physics_model(config)
qoi = get_qoi(config)
downsampler = get_downsampler(config)
nn_model = get_nn_model(config)

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

train_dataset = list(itertools.islice(dataset, 0, n_samples_train))
valid_dataset = list(
    itertools.islice(dataset, n_samples_train, n_samples_train + n_samples_valid)
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config["train"]["batch_size"], shuffle=True
)
valid_dataloader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=n_samples_valid
)


physics_model_lowres = physics_model_highres.coarsen(
    config["discretisation"]["scaling_factor"]
)

model = PEDSModel(physics_model_lowres, downsampler, qoi, nn_model)

n_param = sum([torch.numel(p) for p in model.parameters()])
print(f"number of model parameters = {n_param}")

model = model.to(device)
coarse_model = torch.nn.Sequential(downsampler, physics_model_lowres, qoi)


loss_fn = torch.nn.MSELoss()
gamma = (lr_target / config["train"]["lr_initial"]) ** (1 / config["train"]["n_epoch"])
print(f"learning rate decay factor = {gamma:8.5f}")
optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr_initial"])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

writer = SummaryWriter(flush_secs=5)
print(f"epoch      :  training    validation    coarse")
for epoch in range(config["train"]["n_epoch"]):
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
        train_loss_avg += train_loss / (n_samples_train / config["train"]["batch_size"])
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
    writer.add_scalar("NN weight", model.w.detach(), epoch)
    writer.add_scalar("gain", coarse_loss / valid_loss, epoch)
    writer.add_scalar("learning rate", scheduler.get_last_lr()[0], epoch)
    print(
        f"epoch {epoch+1:5d}:  {train_loss_avg:12.6f} {valid_loss:12.6f} {coarse_loss:12.6f}"
    )
    scheduler.step()

writer.flush()

model.save(config["model"]["filename"])
