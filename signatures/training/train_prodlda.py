import logging
from typing import Any

import torch
import torch.utils
import torch.utils.data
import wandb
from tqdm import tqdm

from signatures import config
from signatures.models import losses, prodlda
from signatures.utils import data_utils


logging.basicConfig(level=logging.INFO)


def train_prodlda(mutations_df_path: str, training_config: dict[str, Any]):
    wandb.init(
        project=config.PROJECT_NAME,
        config=training_config
    )
    torch.manual_seed(config.RANDOM_SEED)
    logging.info(f"Initialized Weights & Biases experiment. Torch seed set to {config.RANDOM_SEED}.")

    num_mutation_types = len(config.COSMIC_MUTATION_TYPES)
    mutations_dataset = data_utils.MutationsDataset(
        mutations_df_path, 
        training_config["augmentation"], 
        num_mutation_types
    )
    data_loader = torch.utils.data.DataLoader(
        mutations_dataset, 
        batch_size=training_config["batch_size"], 
        shuffle=True,
        pin_memory=True
    )
    logging.info(f"Loaded mutations dataset; shape of the observation matrix: {mutations_dataset.mutation_counts.shape}")

    if not torch.cuda.is_available():
        logging.warning("CUDA is not available, using CPU; this may be very slow!")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = prodlda.ProdLDA(
        vocab_size=num_mutation_types,
        num_topics=training_config["num_topics"], 
        hidden_size=training_config["hidden_size"], 
        dropout=training_config["dropout"],
        device=device
    )
    model.to(device)
    wandb.watch(model, log="all")
    logging.info(f"Initialized ProdLDA model with {training_config['num_topics']} topics.")

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=training_config["lr"],
        betas=training_config["optimizer_betas"]
    )
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=445, epochs=100)

    loss_history, nll_history, kld_history = [], [], []

    logging.info(f"Starting training for {training_config['num_epochs']} epochs.")
    # torch.autograd.set_detect_anomaly(True)

    for epoch in range(training_config["num_epochs"]):
        logging.info(f"Epoch {epoch + 1}/{training_config['num_epochs']}")
        model.train()
        running_loss = running_nll = running_kld = 0.0
        annealing_steps = int(training_config["kl_annealing"]) or 0
        disable_annealing = training_config["kl_annealing"] is None
        kl_base_rate = training_config["kl_base_rate"]
        annealer = losses.Annealer(total_steps=annealing_steps, shape="cosine", baseline=kl_base_rate, cyclical=False, disable=disable_annealing)

        for batch in tqdm(data_loader):
            batch = batch.float().to(device)
            outputs, posterior = model(batch)
            nll, kld = losses.variational_loss(
                batch, outputs, posterior, training_config["kl_beta"]
            )
            kld = annealer(kld)
            loss = nll + kld
            optimizer.zero_grad()
            loss.backward()

            if training_config["clip_norm"] is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=training_config["clip_norm"], 
                    norm_type=2
                )
            optimizer.step()
            # scheduler.step()

            running_loss += loss.item()
            running_nll += nll.item()
            running_kld += kld.item()
        
        annealer.step()
        wandb.log({
            "train/total_loss": running_loss,
            "train/reconstruction_loss": running_nll,
            "train/KL_divergence": running_kld,
            "train/lr": optimizer.param_groups[0]["lr"]
        })
        logging.info(f"Running loss: {running_loss:.3f}; NLL: {running_nll:.3f}; KLD: {running_kld:.3f}")
        loss_history.append(running_loss)
        nll_history.append(running_nll)
        kld_history.append(running_kld)

    wandb.finish()
    return model, loss_history, nll_history, kld_history
