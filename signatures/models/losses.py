import torch
import numpy as np
from torch import nn
from torch import distributions
from torch.nn import functional as F
import torch.utils


def reconstruction_loss(targets: torch.Tensor, outputs:torch.Tensor) -> torch.Tensor:
    return F.mse_loss(outputs, targets)  # Mean squared error
    # return F.nll_loss(outputs, targets, reduction="sum")  # Negative log-likelihood; reduction: sum
    # return -torch.sum(targets*outputs)  # Negative log-likelihood; reduction: mean


def kld_loss(posterior: distributions.Distribution, prior: distributions.Distribution) -> torch.Tensor:
    return distributions.kl_divergence(posterior, prior).sum()


def variational_loss(
        inputs: torch.Tensor, 
        model_outputs: nn.Module, 
        posterior: distributions.Distribution,
        kl_beta: float = 1.) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Variational loss for ProdLDA VAE.
    
    Reconstruction component: negative log-likelihood; probabilistic component: Kullback-Leibler divergence.

    Args:
        inputs (torch.Tensor): _description_
        model (torch.Module): _description_
        kl_beta (float): weight of the K-L component of the loss. Defaults to 1.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: reconstruction loss, K-L divergence.
    """
    # nll = reconstruction_loss(inputs, model_outputs) * nll_weight
    reconstruction_loss = F.mse_loss(model_outputs, inputs)
    prior = standard_prior_like(posterior)
    kld = distributions.kl_divergence(posterior, prior).sum() * kl_beta
    return reconstruction_loss, kld


def standard_prior_like(posterior: distributions.Distribution) -> distributions.Distribution:
    loc = torch.zeros_like(posterior.loc)
    scale = torch.ones_like(posterior.scale)
    prior = distributions.LogNormal(loc, scale)
    return prior


class Annealer:
    """
    This class is used to anneal the KL divergence loss over the course of training VAEs.
    After each call, the step() function should be called to update the current epoch.
    """

    def __init__(self, total_steps, shape, baseline=0.0, cyclical=False, disable=False):
        """
        Parameters:
            total_steps (int): Number of epochs to reach full KL divergence weight.
            shape (str): Shape of the annealing function. Can be 'linear', 'cosine', or 'logistic'.
            baseline (float): Starting value for the annealing function [0-1]. Default is 0.0.
            cyclical (bool): Whether to repeat the annealing cycle after total_steps is reached.
            disable (bool): If true, the __call__ method returns unchanged input (no annealing).
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.cyclical = cyclical
        self.shape = shape
        self.baseline = baseline
        if disable:
            self.shape = 'none'
            self.baseline = 0.0

    def __call__(self, kld):
        """
        Args:
            kld (torch.tensor): KL divergence loss
        Returns:
            out (torch.tensor): KL divergence loss multiplied by the slope of the annealing function.
        """
        out = kld * self.slope()
        return out

    def slope(self):
        if self.shape == 'linear':
            y = (self.current_step / self.total_steps)
        elif self.shape == 'cosine':
            y = (np.cos(np.pi * (self.current_step / self.total_steps - 1)) + 1) / 2
        elif self.shape == 'logistic':
            exponent = ((self.total_steps / 2) - self.current_step)
            y = 1 / (1 + np.exp(exponent))
        elif self.shape == 'none':
            y = 1.0
        else:
            raise ValueError('Invalid shape for annealing function. Must be linear, cosine, or logistic.')
        y = self.add_baseline(y)
        return y

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0
        return

    def add_baseline(self, y):
        y_out = y * (1 - self.baseline) + self.baseline
        return y_out

    def cyclical_setter(self, value):
        if value is not bool:
            raise ValueError('Cyclical_setter method requires boolean argument (True/False)')
        else:
            self.cyclical = value
        return
