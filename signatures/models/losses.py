import torch
from torch import distributions


def reconstruction_loss(targets: torch.Tensor, outputs:torch.Tensor) -> torch.Tensor:
    return -torch.sum(targets*outputs)  # Negative log-likelihood


def kld_loss(posterior: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    return torch.sum(distributions.kl_divergence(posterior, prior))
