from typing import Any

import pandas as pd
from torch.utils.data import Dataset


class DenoisingDataset(Dataset):
    """Returns a dataset of noisy and clean mutation data. When iterating, returns a tuple of (noisy, clean)."""
    def __init__(self, noisy_df_path: str, clean_df_path: str, num_mutation_types: int = 96) -> None:
        super().__init__()
        self.noisy_data = pd.read_csv(noisy_df_path).T.values  # shape: (n_examples, n_mutation_types)
        self.clean_data = pd.read_csv(clean_df_path).T.values  # shape: (n_examples, n_mutation_types)
    
    def __len__(self) -> int:
        return self.noisy_data.shape[0]
    
    def __getitem__(self, idx) -> Any:
        return self.noisy_data[idx, :], self.clean_data[idx, :]
