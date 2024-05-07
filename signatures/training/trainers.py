import logging

import numpy as np
import pandas as pd
import polars as pl
import pyro
import torch
import tqdm
from pyro.infer import SVI, TraceMeanField_ELBO

from signatures import config
from signatures.models import prodlda_pyro


logging.basicConfig(level=logging.INFO)


def train_prodLDA(mutations_df: pd.DataFrame|np.ndarray) -> tuple[prodlda_pyro.ProdLDAPyro, list[float]]:
    """Train a ProdLDA model on the given data.

    Args:
        counts (pd.DataFrame|pl.DataFrame): A dataframe of counts data.
        n_signatures (int): The number of signatures to train.

    Returns:
        None
    """
    mutation_counts = mutations_df.to_numpy() if isinstance(mutations_df, pd.DataFrame) else mutations_df
    if mutation_counts.shape[1] != len(config.COSMIC_MUTATION_TYPES):
        mutation_counts = mutation_counts.transpose()
    mutation_counts = torch.from_numpy(mutation_counts)  # expected shape: (n_samples, n_features), e.g. (569, 96)
    assert mutation_counts.shape[1] == len(config.COSMIC_MUTATION_TYPES), f"Expected {len(config.COSMIC_MUTATION_TYPES)} mutations, got {mutation_counts.shape[1]}"

    torch.manual_seed(config.RANDOM_SEED)
    pyro.set_rng_seed(config.RANDOM_SEED)

    if not torch.cuda.is_available():
        logging.warning("CUDA is not available, using CPU; this may be very slow!")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mutation_counts = mutation_counts.float().to(device)

    pyro.clear_param_store()

    prodLDA = prodlda_pyro.ProdLDAPyro(
        vocab_size=mutation_counts.shape[1],
        num_topics=config.N_SIGNATURES_TARGET,
        hidden=config.HIDDEN_SIZE,
        dropout=config.DROPOUT,
        loss_regularizer=config.LOSS_REGULARIZER,
        reg_lambda=config.REGULARIZER_LAMBDA
    )
    prodLDA.to(device)

    optimizer = pyro.optim.Adam({"lr": config.LEARNING_RATE, "betas": config.BETAS})
    svi = SVI(prodLDA.model, prodLDA.guide, optimizer, loss=TraceMeanField_ELBO())
    num_batches = int(np.ceil(mutation_counts.shape[0] / config.BATCH_SIZE))

    bar = tqdm.trange(config.NUM_EPOCHS)
    epoch_loss = []
    for _ in bar:
        running_loss = 0.0
        for i in range(num_batches):
            batch_counts = mutation_counts[i * config.BATCH_SIZE:(i + 1) * config.BATCH_SIZE, :]
            loss = svi.step(batch_counts)
            running_loss += loss / batch_counts.size(0)
        epoch_loss.append(running_loss)
        bar.set_postfix(epoch_loss='{:.2e}'.format(running_loss))
    
    return prodLDA, epoch_loss