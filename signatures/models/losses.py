import torch
from torch import nn
from torch import distributions
from torch.nn import functional as F
import torch.utils


def reconstruction_loss(targets: torch.Tensor, outputs:torch.Tensor) -> torch.Tensor:
    # return F.mse_loss(outputs, targets, reduction="sum")  # Mean squared logarithmic error
    # return F.nll_loss(outputs, targets, reduction="sum")  # Negative log-likelihood; reduction: sum
    return -torch.sum(targets*outputs)  # Negative log-likelihood; reduction: mean


def kld_loss(posterior: distributions.Distribution, prior: distributions.Distribution) -> torch.Tensor:
    return distributions.kl_divergence(posterior, prior).sum()


def variational_loss(
        inputs: torch.Tensor, 
        model_outputs: nn.Module, 
        posterior: distributions.Distribution,
        nll_weight: float = 1.,
        kl_weight: float = 1.) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Variational loss for ProdLDA VAE.
    
    Reconstruction component: negative log-likelihood; probabilistic component: Kullback-Leibler divergence.

    Args:
        inputs (torch.Tensor): _description_
        model (torch.Module): _description_
        nll_weight (float): weight of the reconstruction component of the loss. Defaults to 1.
        kl_weight (float): weight of the K-L component of the loss. Defaults to 1.
        device (_type_, optional): device. Defaults to "cuda:0".

    Returns:
        tuple[torch.Tensor, torch.Tensor]: reconstruction loss, K-L divergence.
    """
    prior = standard_prior_like(posterior)
    nll = reconstruction_loss(inputs, model_outputs) * nll_weight
    kld = kld_loss(posterior, prior) * kl_weight
    return nll, kld


def standard_prior_like(posterior: distributions.Distribution) -> distributions.Distribution:
    loc = torch.zeros_like(posterior.loc)
    scale = torch.ones_like(posterior.scale)
    prior = distributions.LogNormal(loc, scale)
    return prior
