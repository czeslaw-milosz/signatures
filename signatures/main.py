
"""Placeholder for the main script of the project."""
import logging

import pandas as pd
import torch
from torch.nn import functional as F
from matplotlib import pyplot as plt
from signatures import config
from signatures.evaluation import evaluation
from signatures.utils import data_utils
from signatures.training import trainers


logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    cosmic = data_utils.read_cosmic_signatures()
    logging.info(f"Loaded cosmic signatures; shape: {cosmic.shape}")
    mutation_counts = data_utils.load_mutation_counts()
    if config.APPLY_AUGMENTATION:
        mutation_counts = data_utils.augment_data(mutation_counts)
    logging.info(f"Loaded mutation counts; shape: {mutation_counts.shape}")
    # mutation_counts = data_utils.normalize(mutation_counts)

    model, loss_history = trainers.train_prodLDA(mutations_df=mutation_counts)

    plt.plot(loss_history)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("loss history")
    plt.savefig(config.LOSS_PLOT_PATH)
    if config.SAVE_MODEL:
        torch.save(model.state_dict(), config.PRODLDA_MODEL_PATH)

    P = model.beta()
    # P_prob = (P / P.sum(dim=1).unsqueeze(-1)).numpy()
    P_prob = F.softmax(P, dim=1).detach().numpy()
    nearest_signatures = evaluation.get_nearest_signatures(P_prob, cosmic)
    print(list(nearest_signatures.items()))
    discovered_signatures = pd.DataFrame(P_prob.transpose(), index=pd.Index(data=cosmic.index, name="Type"))
    discovered_signatures.to_csv(f"resources/{config.EXPERIMENT_NAME}_discovered_signatures.csv")
