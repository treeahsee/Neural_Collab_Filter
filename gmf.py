import torch
from torch import nn

class GMF(torch.nn.Module):
    def __init__(self, num_users, num_items, embed_dim, top_depth=0):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim

        self.user_embedding = nn.Embedding(self.num_users, self.embed_dim)
        self.movie_embedding = nn.Embedding(self.num_items, self.embed_dim)
        
        self.linear = nn.Linear(self.embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

        # Used for Neural MF
        self.top_depth = top_depth

    def forward(self, users, movies):
        gmf = self.user_embedding(users) * self.movie_embedding(movies)
        # print('gmf', self.top_depth)

        # Last nonlinearity is not used in Neural MF
        if self.top_depth == 2:
            return gmf

        x = self.linear(gmf)
        if self.top_depth == 1:
            return x

        # return x
        return self.sigmoid(x)
