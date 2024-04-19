import torch
from torch import nn

class GMF(torch.nn.Module):
    def __init__(self, num_users, num_movies, embed_dim):
        super().__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.embed_dim = embed_dim

        self.user_embedding = nn.Embedding(self.num_users, self.embed_dim)
        self.movie_embedding = nn.Embedding(self.num_movies, self.embed_dim)
        
        self.linear = nn.Linear(self.embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, users, movies):
        """
        :param x: Long tensor of size ``(batch_size, num_user_fields)``
        """
        # TODO: Consider not hard-coding user idx
        # users = users_and_movies[:, 0]
        # movies = users_and_movies[:, 1]
        
        gmf = self.user_embedding(users) * self.movie_embedding(movies)
        x = self.linear(gmf)
        return self.sigmoid(x)
