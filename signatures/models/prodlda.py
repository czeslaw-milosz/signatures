import torch
from torch import distributions
from torch import nn
from torch.nn import functional as F


class ProdLDA(nn.Module):
    def __init__(self, vocab_size: int, num_topics: int, hidden_size: int, dropout: tuple[float], device: str = "cuda:0") -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.hidden_size = hidden_size
        try:
            self.encoder_dropout, self.decoder_dropout = dropout
        except:
            self.encoder_dropout = self.decoder_dropout = dropout[0]
        self.encoder = ProdLDAEncoder(vocab_size, num_topics, hidden_size, self.encoder_dropout)
        self.decoder = ProdLDADecoder(vocab_size, num_topics, self.decoder_dropout)
        self.device = device
    

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, distributions.Distribution]:
        posterior = self.encoder(inputs)
        if self.training:
            z = posterior.rsample().to(inputs.device)
        else:
            z = posterior.mean.to(inputs.device)
        z /= z.sum(1, keepdim=True)
        outputs = self.decoder(z)
        return outputs, posterior


class ProdLDAEncoder(nn.Module):
    def __init__(self, vocab_size: int, num_topics: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(vocab_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, num_topics)  # mean of the log-normal
        self.fc_logvar = nn.Linear(hidden_size, num_topics)  # log-variance of the log-normal
        self.bn_mu = nn.BatchNorm1d(num_topics, affine=False)
        self.bn_logvar = nn.BatchNorm1d(num_topics, affine=False)

    def forward(self, inputs: torch.Tensor) -> distributions.Distribution:
        """Forward pass of the encoder:

        Args:
            inputs (torch.Tensor): _description_

        Returns:
            distributions.Distribution: log-normal distribution with parameters mu, exp(logvar/2) determined by the network.
        """
        h1 = F.softplus(self.fc1(inputs))
        h2 = F.softplus(self.fc2(h1))
        mu = self.bn_mu(self.fc_mu(h2))
        logvar = self.bn_logvar(self.fc_logvar(h2))
        dist = distributions.LogNormal(mu, (0.5 * logvar).exp())  # reparametrization; using exp(log(var)) enforces positivity
        return dist
    

class ProdLDADecoder(nn.Module):
    def __init__(self, vocab_size: int, num_topics: int, dropout: float, use_batch_norm: bool = True) -> None:
        super().__init__()
        self.beta = nn.Linear(num_topics, vocab_size, bias=False)
        self.bn = nn.BatchNorm1d(vocab_size, affine=False) if use_batch_norm else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the encoder: single linear layer (beta = unnormalized signatures matrix) with softmax.

        Args:
            inputs (torch.Tensor): _description_

        Returns:
            torch.Tensor: should be the reconstructed matrix of observed counts.
        """        
        inputs = self.dropout(inputs)
        # the output is σ(βθ)
        outputs = self.beta(inputs) if self.bn is None else self.bn(self.beta(inputs))
        return F.softmax(outputs, dim=1)
