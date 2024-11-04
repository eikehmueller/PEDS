import itertools
import torch
from torch.utils.tensorboard import SummaryWriter

from peds.diffusion_model import DiffusionModel1d
from peds.distributions import LogNormalDistribution1d
from peds.quantity_of_interest import QoISampling1d
from peds.datasets import PEDSDataset
from peds.peds_model import PEDSModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on device {device}")

use_peds = True  # use PEDS model?
n = 256  # number of grid cells
Lambda = 0.1  # correlation length
a_power = 2  # power in log-normal distribution
n_samples_train = 2048  # number of training samples
n_samples_valid = 32  # number of validation samples
batch_size = 32  # batch size
n_epoch = 1000
scaling_factor = 8  # ratio between fine and coarse grid cells
n_lowres = n // scaling_factor  # number of cells of fine grid

sample_points = [0.1, 0.3, 0.5, 0.7, 0.9]

f_rhs = torch.ones(size=(n,), dtype=torch.float)

distribution = LogNormalDistribution1d(n, Lambda, a_power)
physics_model_highres = DiffusionModel1d(f_rhs)
qoi = QoISampling1d(sample_points)

dataset = PEDSDataset(distribution, physics_model_highres, qoi)
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
    torch.nn.AvgPool1d(1, stride=scaling_factor),
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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

writer = SummaryWriter(flush_secs=5)
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
