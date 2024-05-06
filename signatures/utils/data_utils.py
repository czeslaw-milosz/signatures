import numpy as np
import pandas as pd

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



