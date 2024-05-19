from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from signatures import config


def read_cosmic_signatures(input_path: str = config.COSMIC_FILE_PATH) -> pd.DataFrame:
    return pd.read_csv(input_path, sep="\t").set_index("Type")


def load_mutation_counts(input_path: str = config.BRCA_COUNTS_FILE_PATH) -> pd.DataFrame:
    return pd.read_csv(
        input_path, index_col=0
    ).rename_axis("Type", axis=0).T


def augment_data(X: pd.DataFrame | np.ndarray, augmentation_factor: int = 100) -> np.ndarray:
    X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
    gen = np.random.default_rng(config.RANDOM_SEED)
    return np.vstack([
        bootstrap_multinomial(x, sample_size=augmentation_factor, random_generator=gen)
        for x in X
    ])


def bootstrap_multinomial(X: np.ndarray, sample_size: int, random_generator: np.random.Generator | None = None) -> np.ndarray:
    random_generator = np.random.default_rng() if random_generator is None else random_generator
    N = int(round(np.sum(X)))
    p = np.ravel(X/np.sum(X))
    return random_generator.multinomial(N, p, size=sample_size)


def normalize(X: np.ndarray) -> np.ndarray:
    return X/X.sum(axis=1, keepdims=True) * np.log2(X.sum(axis=1, keepdims=True))


class MutationsDataset(Dataset):
    def __init__(self, mutations_df_path: str, augmentation_factor: int = 100, num_mutation_types: int = 96) -> None:
        super().__init__()
        self.mutation_counts = load_mutation_counts(mutations_df_path)
        assert augmentation_factor >= 0, f"Misspecified augmentation factor {augmentation_factor} < 0"
        self.augmentation_factor = augmentation_factor
        if self.augmentation_factor:
            self.mutation_counts = augment_data(self.mutation_counts)
            self.mutation_counts = normalize(self.mutation_counts)
        if isinstance(self.mutation_counts, pd.DataFrame):
            self.mutation_counts = self.mutation_counts.to_numpy()
        if self.mutation_counts.shape[1] != num_mutation_types:
            self.mutation_counts = self.mutation_counts.transpose()  # expected shape: (n_samples, n_features), e.g. (569, 96)
        assert self.mutation_counts.shape[1] == num_mutation_types, f"Expected {num_mutation_types} mutations, got {self.mutation_counts.shape[1]}"
    
    def __len__(self) -> int:
        return self.mutation_counts.shape[0]
    
    def __getitem__(self, idx) -> Any:
        return self.mutation_counts[idx, :]
