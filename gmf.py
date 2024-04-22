import torch
from torch import nn

class GMF(torch.nn.Module):
    def __init__(self, num_users, num_items, embed_dim, output_top=True):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim

        self.user_embedding = nn.Embedding(self.num_users, self.embed_dim)
        self.movie_embedding = nn.Embedding(self.num_items, self.embed_dim)
        
        self.linear = nn.Linear(self.embed_dim, 1)
        # self.sigmoid = nn.Sigmoid()

        # Used for Neural MF
        self.output_top = output_top

    def forward(self, users, movies):
        gmf = self.user_embedding(users) * self.movie_embedding(movies)

        # Last nonlinearity is not used in Neural MF
        if not self.output_top:
            return gmf

        x = self.linear(gmf)

        # return self.sigmoid(x)
        return x
