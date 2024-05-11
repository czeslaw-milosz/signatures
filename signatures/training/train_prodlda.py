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

    num_mutation_types = len(config.COSMIC_MUTATION_TYPES)
    mutations_dataset = data_utils.MutationsDataset(
        mutations_df_path, 
        training_config["augmentation_factor"], 
        num_mutation_types
    )
    data_loader = torch.utils.data.DataLoader(
        mutations_dataset, 
        batch_size=training_config["batch_size"], 
        shuffle=True,
        pin_memory=True
    )
    # mutation_counts = mutations_df.to_numpy() if isinstance(mutations_df, pd.DataFrame) else mutations_df
    # num_mutation_types = len(config.COSMIC_MUTATION_TYPES)
    # if mutation_counts.shape[1] != num_mutation_types:
    #     mutation_counts = mutation_counts.transpose()
    # mutation_counts = torch.from_numpy(mutation_counts)  # expected shape: (n_samples, n_features), e.g. (569, 96)
    # assert mutation_counts.shape[1] == num_mutation_types, f"Expected {num_mutation_types} mutations, got {mutation_counts.shape[1]}"

    if not torch.cuda.is_available():
        logging.warning("CUDA is not available, using CPU; this may be very slow!")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # mutation_counts = mutation_counts.float().to(device)

    model = prodlda.ProdLDA(
        vocab_size=num_mutation_types,
        num_topics=training_config["num_topics"], 
        hidden_size=training_config["hidden_size"], 
        dropout=training_config["dropout"],
        device=device
    )
    model.to(device)
    wandb.watch(model)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=training_config["learning_rate"],
        betas=training_config["optimizer_betas"]
    )

    loss_history, nll_history, kld_history = [], [], []

    for epoch in range(training_config["num_epochs"]):
        logging.info(f"Epoch {epoch + 1}/{training_config['num_epochs']}")
        model.train()
        running_loss = running_nll = running_kld = 0.0

        for batch in tqdm(data_loader):
            optimizer.zero_grad()
            outputs, posterior = model(batch)
            nll, kld = losses.variational_loss(
                batch, outputs, posterior, training_config["nll_weight"], training_config["kl_weight"], device
            )
            loss = nll + kld
            loss.backward()
            optimizer.step()
            running_loss += loss.item() / len(batch)
            running_nll += nll.item() / len(batch)
            running_kld += kld.item() / len(batch)

        wandb.log({
            "train/total_loss": loss,
            "train/reconstruction_loss": nll,
            "train/KL_divergence": kld,
        })
        logging.info(f"Running loss: {running_loss:.3f}; NLL: {running_nll:.3f}; KLD: {running_kld:.3f}")
        loss_history.append(running_loss)
        nll_history.append(running_nll)
        kld_history.append(running_kld)

    wandb.finish()
    return model, loss_history, nll_history, kld_history
