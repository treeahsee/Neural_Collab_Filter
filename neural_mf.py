import torch
from torch import nn

from gmf import GMF
from ncf_mlp import NCF_MLP as MLP

class NEURAL_MF(torch.nn.Module):
    def __init__(self, num_users, num_items, latent_dims, gmf=None, mlp=None, alpha=0.5):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dims = latent_dims

        # By instantiating each of these distinctly, use distinct embeddings for each
        if gmf is None:
            self.gmf = GMF(self.num_users, self.num_items, self.latent_dims, output_top=True)
        else:
            self.gmf = gmf
        if mlp is None:
            self.mlp = MLP(self.num_users, self.num_items, self.latent_dims, output_top=True)
        else:
            self.mlp = mlp
        
        # Ensure that regardless of how the GMF, MLP were trained, we don't use their
        # nonlinear activations for the forward pass anymore.
        self.gmf.output_top = False
        self.mlp.output_top = False

        with torch.no_grad():
            self.gmf.user_embedding.weight = nn.Parameter(self.gmf.user_embedding.weight * alpha)
            self.gmf.movie_embedding.weight = nn.Parameter(self.gmf.movie_embedding.weight * alpha)
            self.mlp.l3.weight = nn.Parameter(self.mlp.l3.weight * (1 - alpha))

        gmf_vector_length = self.gmf.embed_dim
        mlp_vector_length = self.mlp.l4.weight.shape[1] # Input dims to last layer

        concat_length = gmf_vector_length + mlp_vector_length

        self.linear = nn.Linear(concat_length, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, users, movies):
        gmf_top = self.gmf(users, movies)
        mlp_top = self.mlp(users, movies)

        top_outputs = torch.cat([gmf_top, mlp_top], dim=1)

        x = self.linear(top_outputs)
        return self.sigmoid(x)
