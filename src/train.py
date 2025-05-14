import os
import torch
from torch.utils.tensorboard import SummaryWriter

from peds.datasets import PEDSDataset
from peds.peds_model import PEDSModel
from setup import (
    read_config,
    get_distribution,
    get_physics_model,
    get_qoi,
    get_downsampler,
    get_nn_model,
    get_pure_nn_model,
    get_datasets,
)

if __name__ == "__main__":

    config = read_config()

    device = torch.device(
        "cuda:0"
        if config["model"]["dimension"] == 1 and torch.cuda.is_available()
        else "cpu"
    )

    print(f"Running on device {device}")

    distribution = get_distribution(config)
    physics_model_highres = get_physics_model(config)
    qoi = get_qoi(config)
    downsampler = get_downsampler(config)
    nn_model = get_nn_model(config)

    if not os.path.exists(config["data"]["filename"]):
        dataset = PEDSDataset(distribution, physics_model_highres, qoi)
        dataset.save(
            config["data"]["n_samples_train"]
            + config["data"]["n_samples_valid"]
            + config["data"]["n_samples_test"],
            config["data"]["filename"],
        )
    train_dataset, valid_dataset, _ = get_datasets(config)

    n_samples_train = config["data"]["n_samples_train"]
    n_samples_valid = config["data"]["n_samples_valid"]

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

    model.to(device)
    coarse_model = torch.nn.Sequential(downsampler, physics_model_lowres, qoi)

    loss_fn = torch.nn.MSELoss()
    gamma = (config["train"]["lr_final"] / config["train"]["lr_initial"]) ** (
        1 / config["train"]["n_epoch"]
    )
    print(f"learning rate decay factor = {gamma:8.5f}")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr_initial"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    writer = SummaryWriter(flush_secs=5)
    if config["train"]["train_peds"]:
        ##### Train PEDS model ####
        print("epoch      :  training    validation    coarse")
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
                train_loss_avg += train_loss / (
                    n_samples_train / config["train"]["batch_size"]
                )
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
        model.to("cpu")
        model.save(config["model"]["peds_filename"])

    ##### Train pure NN model ####

    pure_nn_model = get_pure_nn_model(config)

    n_param = sum([torch.numel(p) for p in pure_nn_model.parameters()])
    print(f"number of model parameters = {n_param}")

    pure_nn_model.to(device)

    optimizer = torch.optim.Adam(pure_nn_model.parameters(), lr=config["train"]["lr_initial"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    writer = SummaryWriter(flush_secs=5)

    if config["train"]["train_pure_nn"]:
        ##### Train pure NN model ####
        print("epoch      :  training    validation    coarse")
        for epoch in range(config["train"]["n_epoch"]):
            train_loss_avg = 0
            for i, data in enumerate(train_dataloader):
                alpha, q_target = data
                alpha = alpha.to(device)
                q_target = q_target.to(device)
                optimizer.zero_grad()
                q_pred = pure_nn_model(alpha)
                loss = loss_fn(q_pred, q_target)
                loss.backward()
                optimizer.step()
                train_loss = loss.item()
                train_loss_avg += train_loss / (
                    n_samples_train / config["train"]["batch_size"]
                )
            alpha, q_target = next(iter(valid_dataloader))
            alpha = alpha.to(device)
            q_target = q_target.to(device)
            q_pred = pure_nn_model(alpha)
            valid_loss = loss_fn(q_pred, q_target)
            q_pred_coarse = coarse_model(alpha)
            coarse_loss = loss_fn(q_pred_coarse, q_target)
            writer.add_scalars(
                "Loss",
                {"train": train_loss_avg, "valid": valid_loss, "coarse": coarse_loss},
                epoch,
            )
            writer.add_scalar("gain", coarse_loss / valid_loss, epoch)
            writer.add_scalar("learning rate", scheduler.get_last_lr()[0], epoch)
            print(
                f"epoch {epoch+1:5d}:  {train_loss_avg:12.6f} {valid_loss:12.6f} {coarse_loss:12.6f}"
            )
            scheduler.step()
        pure_nn_model.to("cpu")
        torch.save(pure_nn_model, config["model"]["pure_nn_filename"])
