import torch
from torch import nn

class NCF_MLP(nn.Module):
    def __init__(self, num_users, num_items, latent_dims):
        super().__init__()
        ## num users + num items + latent dims
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dims = latent_dims

        ## User and Items Embeddings
        self.mlp_embed_users = nn.Embedding(self.num_users, self.latent_dims)
        self.mlp_embed_items = nn.Embedding(self.num_items, self.latent_dims)
        
        ##layers
        self.mlp_layers = nn.Sequential(
            nn.Linear(2 * self.latent_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, user, items):

        ## user embedding
        user_embed = self.mlp_embed_users(user)
        ##item embedding
        item_embed = self.mlp_embed_items(items)

        ##concat
        user_item = torch.cat([user_embed, item_embed], dim = 1)

        logits = self.mlp_layers(user_item)
        # pred = torch.Sigmoid(logits)

        return logits