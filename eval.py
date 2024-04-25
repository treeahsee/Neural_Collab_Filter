from tqdm.notebook import tqdm

import torch
from torch import nn
import numpy as np

from heapq import heappush, heappop, nsmallest

class MovieRatingsHeap:
    def __init__(self, k):
        self.k = k
        self.heap = []
        self.key = lambda x: x[1]
    
    def add_rating(self, rating, movie):
        # Make higher rated movies higher priority
        rating_priorty = 1 - rating
        rating_tuple = (rating_priorty, movie)

        heappush(self.heap, rating_tuple)

        # Should only iterate once at most!
        while len(self.heap) > self.k:
            heappop(self.heap)
    
    def get_top_ratings(self):
        if len(self.heap) < self.k:
            movie_ratings = sorted(self.heap)
        else:
            movie_ratings = nsmallest(self.k, self.heap)
        
        # Reconstruct original ratings
        movie_ratings = [(1 - rating, movie) for rating, movie in movie_ratings]

        return movie_ratings


def split_given_size(a, size):
    return np.split(a, np.arange(size,len(a),size))


def hit_rate(model,
             k,
             num_users,
             num_movies,
             data,
             device,
             batch_size=100):
    with torch.no_grad():
        users_to_k_ratings = {}
        hits = 0

        movie_batches = split_given_size(np.arange(num_movies), batch_size)

        for user in tqdm(range(num_users)):
            mheap = MovieRatingsHeap(k)
            for movie_batch in movie_batches:
                user_batch = np.ones_like(movie_batch) * user
                user_tensor = torch.tensor(user_batch).to(device)
                movie_tensor = torch.tensor(movie_batch).to(device)

                predicted_rating_tensor = model.forward(user_tensor, movie_tensor)
                pred = predicted_rating_tensor.cpu().numpy()
                for i in range(len(movie_batch)):
                    movie = movie_batch[i]
                    p = pred[i]
                    mheap.add_rating(p, movie)

            users_to_k_ratings[user] = mheap.get_top_ratings()
            for rating_unused, movie in mheap.get_top_ratings():
                if ((data['user_id'] == user) & (data['movie_id'] == movie)).any():
                    hits += 1
                    break

        hit_rate = hits / num_users
        return (hit_rate, users_to_k_ratings)



def precision_at_k(model,
                   k,
                   num_users,
                   num_movies,
                   data,
                   device,
                   batch_size=100):
    with torch.no_grad():
        users_to_k_ratings = {}
        hits = 0

        movie_batches = split_given_size(np.arange(num_movies), batch_size)

        for user in tqdm(range(num_users)):
            mheap = MovieRatingsHeap(k)
            for movie_batch in movie_batches:
                user_batch = np.ones_like(movie_batch) * user
                user_tensor = torch.tensor(user_batch).to(device)
                movie_tensor = torch.tensor(movie_batch).to(device)

                predicted_rating_tensor = model.forward(user_tensor, movie_tensor)
                pred = predicted_rating_tensor.cpu().numpy()
                for i in range(len(movie_batch)):
                    movie = movie_batch[i]
                    p = pred[i]
                    mheap.add_rating(p, movie)

            users_to_k_ratings[user] = mheap.get_top_ratings()
            for rating_unused, movie in mheap.get_top_ratings():
                if ((data['user_id'] == user) & (data['movie_id'] == movie)).any():
                    hits += 1

        hit_rate = hits / (num_users * k)
        return (hit_rate, users_to_k_ratings)
