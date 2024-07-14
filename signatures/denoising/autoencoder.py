import torch
from torch import nn


class DenseDenoisingAE(nn.Module):

    def __init__(self, input_dim: int = 96, latent_dim: int = 20):
        super(DenseDenoisingAE, self).__init__()

        # [b, 96] => [b, 20]
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU()
        )

        # [b, 20] => [b, 96]
        self.decoder = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            # nn.Sigmoid()
        )


    def forward(self, x):
        return self.decoder(self.encoder(x))
