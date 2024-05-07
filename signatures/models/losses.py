import torch
from torch import distributions


def reconstruction_loss(targets: torch.Tensor, outputs:torch.Tensor) -> torch.Tensor:
    return -torch.sum(targets*outputs)  # Negative log-likelihood


def kld_loss(posterior: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    return torch.sum(distributions.kl_divergence(posterior, prior))


def variational_loss(inputs: torch.Tensor, model: torch.Module, device: str = "cuda:0") -> tuple[torch.Tensor, torch.Tensor]:
    inputs = inputs.to(device)
    outputs, posterior = model(inputs)
    prior = standard_prior_like(posterior)
    nll = reconstruction_loss(inputs, outputs)
    kld = torch.sum(kld_loss(posterior, prior).to(device))
    return nll, kld


def standard_prior_like(posterior: distributions.Distribution):
    loc = torch.zeros_like(posterior.loc)
    scale = torch.ones_like(posterior.scale)
    prior = distributions.LogNormal(loc, scale)
    return prior
