
"""Main script of the project."""
import logging

import click
import pandas as pd
import torch
from torch.nn import functional as F
from matplotlib import pyplot as plt
from signatures import config
from signatures.evaluation import evaluation
from signatures.utils import data_utils
from signatures.denoising import training


logging.basicConfig(level=logging.INFO)


@click.command()
@click.option("--noisy_df_path", default="resources/data/denoising/pcawg_synth/simulated_PCAWG_FFPE_unrepaired.csv", help="Path to noisy data.")
@click.option("--clean_df_path", default="resources/data/denoising/pcawg_synth/simulated_PCAWG_FFPE_repaired.csv", help="Path to clean data.")
@click.option("--experiment_name", default="TEST_DENOISING", help="Name of the current experiment.")
@click.option("--batch_size", default=64, help="Batch size for training.")
@click.option("--num_epochs", default=50, help="Number of epochs to train the model.")
@click.option("--lr", default=1e-3, help="Learning rate for the model.")
@click.option("--dropout", multiple=True, default=(0.2,), help="Dropout rate for the model. Can be a single float or a tuple (encoder_dropout, decoder_dropout).")
@click.option("--save_model", default=False, help="Whether to save the trained model.")
def main(**kwargs):
    experiment_config = {k: v for k, v in kwargs.items() if "path" not in k and "save" not in k}
    model, loss_history = training.train_denoising_AE(kwargs["noisy_df_path"], kwargs["clean_df_path"], experiment_config)

    if config.SAVE_MODEL:
        torch.save(model.state_dict(), config.PRODLDA_MODEL_PATH)

    plt.plot(loss_history)
    plt.savefig(f"resources/experiments/TEST_DENOISING_loss_history.png")


if __name__ == "__main__":
    main()
