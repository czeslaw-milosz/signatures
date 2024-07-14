import logging

import sklearn
import torch
import wandb
from torch import nn
from tqdm import tqdm

from signatures import config
from signatures.denoising import datasets, autoencoder


def train_denoising_AE(
        noisy_df_path: str,
        clean_df_path: str,
        training_config: dict[str, int | float | tuple[int, int] | tuple[float, float]]
    ) -> tuple[torch.nn.Module, list[float], list[float]]:
    torch.manual_seed(config.RANDOM_SEED)
    # logging.info(f"Initialized Weights & Biases experiment. Torch seed set to {config.RANDOM_SEED}.")
    # wandb.init(
    #     project="denoising",
    #     config=training_config
    # )

    denoising_dataset = datasets.DenoisingDataset(noisy_df_path, clean_df_path)
    data_loader = torch.utils.data.DataLoader(
        denoising_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        pin_memory=True
    )
    logging.info(f"Loaded denoising dataset; shape of the observation matrix: {denoising_dataset.noisy_data.shape}")

    if not torch.cuda.is_available():
        logging.warning("CUDA is not available, using CPU; this may be very slow!")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = autoencoder.DenseDenoisingAE()
    model.to(device)
    # wandb.watch(model, log="all")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["lr"]
    )
    criterion = nn.L1Loss()
    loss_history = []
    for epoch in range(training_config["num_epochs"]):
        running_loss = 0.0

        for x, y in tqdm(data_loader):
            optimizer.zero_grad()
            x, y = x.float().to(device), y.float().to(device)
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            # loss_history.append(loss.item())
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{training_config['num_epochs']}: running loss = {running_loss/len(data_loader)}")
        loss_history.append(running_loss/len(data_loader))
        # wandb.log({
        #     "train/running_loss": running_loss,
        # })
    
    # wandb.finish()
    return model, loss_history
