import torch
from torch import nn

class NCF_MLP(nn.Module):
    def __init__(self, num_users, num_items, latent_dims, output_top=True):
        super().__init__()
        ## num users + num items + latent dims
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dims = latent_dims

        ## User and Items Embeddings
        self.mlp_embed_users = nn.Embedding(self.num_users, self.latent_dims)
        self.mlp_embed_items = nn.Embedding(self.num_items, self.latent_dims)

        # Keep track of the last linear layer in order to potentially modify its
        # weights later on
        ##layers
        self.l1 = nn.Linear(2 * self.latent_dims, 32)
        self.l2 = nn.Linear(32, 16)
        self.l3 = nn.Linear(16, 8)
        self.l4 = nn.Linear(8, 1)

        self.nl = nn.ReLU()

        # Used for Neural MF
        self.output_top = output_top

        self.sigmoid = nn.Sigmoid()


    def forward(self, user, items):
        ## user embedding
        user_embed = self.mlp_embed_users(user)
        ##item embedding
        item_embed = self.mlp_embed_items(items)

        ##concat
        user_item = torch.cat([user_embed, item_embed], dim = 1)

        x = self.l1(user_item)
        x = self.nl(x)

        x = self.l2(x)
        x = self.nl(x)

        x = self.l3(x)
        
        # Used for neural MF
        if not self.output_top:
            return x

        x = self.nl(x)

        logits = self.l4(x)

        return self.sigmoid(logits)
