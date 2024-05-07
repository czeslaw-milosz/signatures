import torch
from torch import distributions
from torch import nn
from torch.nn import functional as F


class ProdLDAEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_topics: int, dropout: float) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(vocab_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fcmu = nn.Linear(hidden_size, num_topics)
        self.fclv = nn.Linear(hidden_size, num_topics)
        self.bnmu = nn.BatchNorm1d(num_topics)
        self.bnlv = nn.BatchNorm1d(num_topics)

    def forward(self, inputs) -> distributions.Distribution:
        h1 = F.softplus(self.fc1(inputs))
        h2 = F.softplus(self.fc2(h1))
        mu = self.bnmu(self.fcmu(h2))
        lv = self.bnlv(self.fclv(h2))
        dist = distributions.LogNormal(mu, (0.5 * lv).exp())
        return dist


class ProdLDADecoder(nn.Module):
    def __init__(self, vocab_size: int, num_topics: int, dropout: float) -> None:
        super().__init__()
        self.fc = nn.Linear(num_topics, vocab_size)
        self.bn = nn.BatchNorm1d(vocab_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.drop(inputs)
        return F.log_softmax(self.bn(self.fc(inputs)), dim=1)


class ProdLDA(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_topics: int, dropout: float) -> None:
        super().__init__()
        self.encoder = ProdLDAEncoder(vocab_size, hidden_size, num_topics, dropout)
        self.decoder = ProdLDADecoder(vocab_size, num_topics, dropout)

    def forward(self, inputs) -> tuple[torch.Tensor, torch.Tensor]:
        posterior = self.encoder(inputs)
        if self.training:
            t = posterior.rsample().to(inputs.device)
        else:
            t = posterior.mean.to(inputs.device)
        t = t / t.sum(1, keepdim=True)
        outputs = self.decoder(t)
        return outputs, posterior
