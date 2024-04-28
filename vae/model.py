# adapted from https://avandekleut.github.io/vae/
import torch
import torch.nn as nn

class VariationalEncoder(nn.Module):
    def __init__(self, item_dim, emb_dim, latent_dim, hidden_dim):
        super(VariationalEncoder, self).__init__()
        self.item_embedding = nn.Embedding(emb_dim)
        self.mlp_input = nn.Sequential(
            nn.Linear(item_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mlp_mean = nn.Linear(hidden_dim, latent_dim)
        self.mlp_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.mlp_input(x)
        mu = self.mlp_mean(h)
        sigma = torch.exp(self.mlp_var(h))
        return mu, sigma


class VariationalAutoencoder(nn.Module):
    def __init__(self, item_dim, emb_dim, latent_dim, hidden_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(item_dim, emb_dim, latent_dim, hidden_dim)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()

    def forward(self, x):
        mu, sigma = self.encoder.forward(x)
        z = mu + sigma*self.N.sample(mu.shape)
