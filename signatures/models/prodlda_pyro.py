import pyro
import torch
from pyro import distributions as dist
from torch import nn
from torch.nn import functional as F


class ProdLDAPyro(nn.Module):
    def __init__(self, vocab_size, num_topics, hidden, dropout, loss_regularizer=None, reg_lambda=1e+03) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        assert loss_regularizer in (None, "l1", "l2", "cosine")
        self.loss_regularizer = loss_regularizer
        self.reg_lambda = reg_lambda
        self.encoder = ProdLDAEncoder(vocab_size, num_topics, hidden, dropout)
        self.decoder = ProdLDADecoder(vocab_size, num_topics, dropout)

    def model(self, docs):
        pyro.module("decoder", self.decoder)
        with pyro.plate("documents", docs.shape[0]):
            # Dirichlet prior 𝑝(𝜃|𝛼) is replaced by a Softmax-normal distribution
            logtheta_loc = docs.new_zeros((docs.shape[0], self.num_topics))
            logtheta_scale = docs.new_ones((docs.shape[0], self.num_topics))
            logtheta = pyro.sample(
                "logtheta", dist.Normal(logtheta_loc, logtheta_scale).to_event(1))
            theta = F.softmax(logtheta, -1)

            # conditional distribution of 𝑤𝑛 is defined as 𝑤𝑛|𝛽,𝜃 ~ Categorical(𝜎(𝛽𝜃))
            count_param = self.decoder(theta)
            total_count = int(docs.sum(-1).max())
            pyro.sample(
                'obs',
                dist.Multinomial(total_count, count_param),
                obs=docs
            )

    def guide(self, docs):
        pyro.module("encoder", self.encoder)

        if self.loss_regularizer == "l2":
            pyro.factor("beta_penalty", self.reg_lambda * losses.l2_penalty(self.decoder.beta.weight), has_rsample=True)
        elif self.loss_regularizer == "l1":
            pyro.factor("beta_penalty", self.reg_lambda * losses.l1_penalty(self.decoder.beta.weight), has_rsample=True)
        elif self.loss_regularizer == "cosine":
            pyro.factor("beta_penalty", self.reg_lambda * losses.cosine_penalty(self.decoder.beta.weight), has_rsample=True)

        with pyro.plate("documents", docs.shape[0]):
            # Dirichlet prior 𝑝(𝜃|𝛼) is replaced by a logistic-normal distribution, where μ and Σ are the encoder outputs
            logtheta_loc, logtheta_scale = self.encoder(docs)
            logtheta = pyro.sample(
                "logtheta", dist.Normal(logtheta_loc, logtheta_scale).to_event(1))

    def beta(self):
        # beta matrix elements are the weights of the FC layer on the decoder
        return self.decoder.beta.weight.cpu().detach().T


class ProdLDAEncoder(nn.Module):
    def __init__(self, vocab_size, num_topics, hidden, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)  # to avoid component collapse
        self.fc1 = nn.Linear(vocab_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fcmu = nn.Linear(hidden, num_topics)
        self.fclv = nn.Linear(hidden, num_topics)
        self.bnmu = nn.BatchNorm1d(num_topics, affine=False)  # to avoid component collapse
        self.bnlv = nn.BatchNorm1d(num_topics, affine=False)  # to avoid component collapse

    def forward(self, inputs):
        h = F.softplus(self.fc1(inputs))
        h = F.softplus(self.fc2(h))
        h = self.drop(h)
        # μ and Σ are the outputs
        logtheta_loc = self.bnmu(self.fcmu(h))
        logtheta_logvar = self.bnlv(self.fclv(h))
        logtheta_scale = (0.5 * logtheta_logvar).exp()  # Enforces positivity
        return logtheta_loc, logtheta_scale
    

class ProdLDADecoder(nn.Module):
    def __init__(self, vocab_size, num_topics, dropout):
        super().__init__()
        self.beta = nn.Linear(num_topics, vocab_size, bias=False)
        self.bn = nn.BatchNorm1d(vocab_size, affine=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        inputs = self.drop(inputs)
        # the output is σ(βθ)
        return F.softmax(self.beta(inputs), dim=1)
        # return F.softmax(self.bn(self.beta(inputs)), dim=1)