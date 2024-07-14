
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
from signatures.training import train_prodlda


logging.basicConfig(level=logging.INFO)


@click.command()
@click.option("--cosmic_path", default=config.COSMIC_FILE_PATH, help="Path to the COSMIC signatures file.")
@click.option("--input_path", default=config.BRCA_COUNTS_FILE_PATH, help="Path to the input mutation counts file.")
@click.option("--experiment_name", default=config.EXPERIMENT_NAME, help="Name of the current experiment.")
@click.option("--num_topics", default=12, help="Number of topics to discover.")
@click.option("--augmentation", default=100, help="Augmentation factor for the input data. 0 for no augmentation.")
@click.option("--batch_size", default=64, help="Batch size for training.")
@click.option("--num_epochs", default=50, help="Number of epochs to train the model.")
@click.option("--lr", default=1e-3, help="Learning rate for the model.")
@click.option("--dropout", multiple=True, default=(0.2,), help="Dropout rate for the model. Can be a single float or a tuple (encoder_dropout, decoder_dropout).")
@click.option("--hidden_size", default=128, help="Hidden size of the encoder.")
@click.option("--kl_beta", default=1.0, help="Weight of the KL divergence term in the variational loss.")
@click.option("--kl_annealing", default=None, help="Number of epochs for KL to reach full weight when annealing. None means no annealing.")
@click.option("--kl_base_rate", default=0.00001, help="Initial weight for the KL divergence term (used only when annealing is enabled).")
@click.option("--clip_norm", default=None, help="Value to clip the gradient norm.")
@click.option("--optimizer_betas", multiple=True, default=(0.95, 0.999), help="Betas for the Adam optimizer.")
@click.option("--save_model", default=config.SAVE_MODEL, help="Whether to save the trained model.")
def main(**kwargs):
    experiment_config = {k: v for k, v in kwargs.items() if k not in ("cosmic_path", "input_path")}
    cosmic = data_utils.read_cosmic_signatures(kwargs["cosmic_path"])
    logging.info(f"Loaded cosmic signatures; shape: {cosmic.shape}")
    logging.info(f"Experiment configuration: {experiment_config}")
    model, loss_history, nll_history, kld_history = train_prodlda.train_prodlda(kwargs["input_path"], experiment_config)

    if config.SAVE_MODEL:
        torch.save(model.state_dict(), config.PRODLDA_MODEL_PATH)
    
    P = model.beta()
    P_prob = F.softmax(P, dim=1).detach().cpu().numpy().transpose()
    logging.info(f"Discovered signatures; shape: {P_prob.shape}")
    nearest_signatures = evaluation.get_nearest_signatures(P_prob, cosmic)
    print(sorted(list(nearest_signatures.items()), key=lambda x: x[1][1], reverse=True))
    discovered_signatures = pd.DataFrame(P_prob.transpose(), index=pd.Index(data=cosmic.index, name="Type"))
    discovered_signatures.to_csv(f"resources/experiments/{config.EXPERIMENT_NAME}_discovered_signatures.csv")
    


if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     cosmic = data_utils.read_cosmic_signatures()
#     logging.info(f"Loaded cosmic signatures; shape: {cosmic.shape}")
#     mutation_counts = data_utils.load_mutation_counts()
#     if config.APPLY_AUGMENTATION:
#         mutation_counts = data_utils.augment_data(mutation_counts)
#     logging.info(f"Loaded mutation counts; shape: {mutation_counts.shape}")
#     # mutation_counts = data_utils.normalize(mutation_counts)

#     model, loss_history = trainers.train_prodLDA(mutations_df=mutation_counts)

#     plt.plot(loss_history)
#     plt.xlabel("epoch")
#     plt.ylabel("loss")
#     plt.title("loss history")
#     plt.savefig(config.LOSS_PLOT_PATH)
#     if config.SAVE_MODEL:
#         torch.save(model.state_dict(), config.PRODLDA_MODEL_PATH)

#     P = model.beta()
#     # P_prob = (P / P.sum(dim=1).unsqueeze(-1)).numpy()
#     P_prob = F.softmax(P, dim=1).detach().numpy()
#     nearest_signatures = evaluation.get_nearest_signatures(P_prob, cosmic)
#     print(list(nearest_signatures.items()))
#     discovered_signatures = pd.DataFrame(P_prob.transpose(), index=pd.Index(data=cosmic.index, name="Type"))
#     discovered_signatures.to_csv(f"resources/{config.EXPERIMENT_NAME}_discovered_signatures.csv")
