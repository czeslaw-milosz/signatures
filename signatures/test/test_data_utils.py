import logging

import numpy as np

from signatures.utils import data_utils

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    cosmic = data_utils.read_cosmic_signatures()
    logging.info(f"Loaded cosmic signatures; shape: {cosmic.shape}")

    example_counts = data_utils.load_mutation_counts("./resources/data/xae/Example.csv")
    logging.info(f"Loaded example counts; shape: {example_counts.shape}")
    logging.info(f"# of unique values: {np.unique(example_counts).shape}")

    example_normalized = data_utils.normalize(example_counts.to_numpy())
    logging.info(f"Normalized example counts; shape: {example_normalized.shape}")
    logging.info(f"# of unique values after normalization: {np.unique(example_normalized).shape}")
    logging.info(f"Sum of normalized values (axis=1): {example_normalized.sum(axis=1)}")
    logging.info(f"Sum of normalized values (axis=0): {example_normalized.sum(axis=0)}")
    np.save("./resources/data/Example_normalized.npy", example_normalized)

    logging.info(f"Sum of values before aug (axis=1): {example_counts.to_numpy().sum(axis=1)}")
    logging.info(f"Sum of values before aug (axis=0): {example_counts.to_numpy().sum(axis=0)}")
    augmented = data_utils.normalize(data_utils.augment_data(example_counts.to_numpy(), augmentation_factor=100))
    logging.info(f"Augmented example counts; shape: {augmented.shape}")
    logging.info(f"# of unique values after augmentation: {np.unique(augmented).shape}")
    logging.info(f"Sum of augmented values (axis=1): {augmented.sum(axis=1)}")
