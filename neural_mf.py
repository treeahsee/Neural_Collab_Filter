import torch
from torch import nn

from gmf import GMF
from ncf_mlp import NCF_MLP as MLP

class NEURAL_MF(torch.nn.Module):
    def __init__(self, num_users, num_items, latent_dims):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dims = latent_dims

        # By instantiating each of these distinctly, use distinct embeddings for each
        self.gmf = GMF(self.num_users, self.num_items, self.latent_dims, output_top=True)
        self.mlp = MLP(self.num_users, self.num_items, self.latent_dims, output_top=True)
        
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, users, movies):
        gmf_top = self.gmf(users, movies)
        mlp_top = self.mlp(users, movies)

        top_outputs = torch.cat([gmf_top, mlp_top], dim=1)

        x = self.linear(top_outputs)
        return self.sigmoid(x)
