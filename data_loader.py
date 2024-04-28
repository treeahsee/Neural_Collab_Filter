import numpy as np
import pandas as pd
import os
import urllib.request
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader

def load_data(size, rescale_data=False):
    if not os.path.exists(f'ml-{size}.zip'):
        urllib.request.urlretrieve(f'https://files.grouplens.org/datasets/movielens/ml-{size}.zip', f'ml-{size}.zip')

    with zipfile.ZipFile(f"ml-{size}.zip", "r") as f:
        f.extractall("./")

    if size == '100k':
        data = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
        item_dsc = pd.read_csv('ml-100k/u.item', sep='|', names=['movie_id', 'title', 'release_date', 'video_release', 'IMDB_url', \
                                                                'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', \
                                                                'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', \
                                                                'Musical', 'Mystery', 'Romannce', 'Sci-Fi', 'Thriller', 'War', 'Western'],
                                                                    encoding='iso-8859-1')
    
    elif size == '20m':
        data = pd.read_csv('ml-20m/ratings.csv')
        data = data.rename(columns = {'userId': 'user_id', 'movieId': 'movie_id'})


    # IMPORTANT: For regression tasks we need to rescale the ratings between 0,1
    if rescale_data:
        data['rating'] = data['rating'] / 5.0

    unique_users = data['user_id'].unique()
    unique_movies = data['movie_id'].unique()

    ##zero index for embedding layer
    user_map = {old : new for new, old in enumerate(unique_users)}
    data['user_id'] = data.user_id.map(user_map)
    movie_map = {old : new for new, old in enumerate(unique_movies)}
    data['movie_id'] = data.movie_id.map(movie_map)

    ##user item size
    num_users = unique_users.shape[0]
    num_movies = unique_movies.shape[0]

    return data, num_users, num_movies


def train_test_split(data, test_size = 0.2):
    user_group = data.groupby('user_id')

    train, test = [], []

    for user_id, user_group in user_group:
        num_ratings = user_group.shape[0]
        
        #split test size out from each user
        train_count = int(num_ratings * (1-test_size))
        train_ratings = user_group[:train_count]
        test_ratings = user_group[train_count:]

        train.append(train_ratings)
        test.append(test_ratings)

    train_df = pd.concat(train).reset_index()
    test_df = pd.concat(test).reset_index()

    return train_df, test_df


def train_test_split_shuffle(data, test_size=0.2):
    user_group = data.groupby('user_id')

    train, test = [], []

    for user_id, user_group in user_group:

        # split test size out from each user
        # https://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas
        train_ratings = user_group.sample(frac=0.8)
        test_ratings = user_group.drop(train_ratings.index).sample(frac=1.0)

        train.append(train_ratings)
        test.append(test_ratings)

    train_df = pd.concat(train).reset_index()
    test_df = pd.concat(test).reset_index()

    return train_df, test_df


class MovielensDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        return self.users.shape[0]
    
    def __getitem__(self, index):
        return self.users[index], self.movies[index], self.ratings[index]


class MovielensConcatDataset(torch.utils.data.Dataset):
    def __init__(self, dataset1, dataset2, num_users, num_items):
        self.dataset1 = dataset1.groupby("user_id", as_index=False).agg(list)
        self.dataset2 = dataset2.groupby("user_id", as_index=False).agg(list)
        self.num_users = num_users
        self.num_items = num_items

    def __getitem__(self, i):
        entry1 = self.dataset1.iloc[i]
        entry2 = self.dataset2.iloc[i]
        user_id1 = torch.tensor(entry1["user_id"])
        movie_id1 = torch.tensor(entry1["movie_id"])
        rating1 = torch.tensor(entry1["rating"])
        user_id2 = torch.tensor(entry2["user_id"])
        movie_id2 = torch.tensor(entry2["movie_id"])
        rating2 = torch.tensor(entry2["rating"])
        return (user_id1,
                torch.sparse_coo_tensor(movie_id1.unsqueeze(0), rating1, size=(self.num_items,)),
                torch.sparse_coo_tensor(movie_id2.unsqueeze(0), rating2, size=(self.num_items,)),)

    def __len__(self):
        return self.num_users

    @staticmethod
    def sparse_collate(batch):
        user_ids, ratings1, ratings2 = zip(*batch)
        return user_ids, torch.stack(ratings1).coalesce(), torch.stack(ratings2).coalesce()


class MovielensAllMovieRatingsPerUserDataset(Dataset):
    def __init__(self, data, num_users, num_items, device="cpu"):
        self.data = data.groupby("user_id", as_index=False).agg(list)
        self.device = device
        self.num_users = num_users
        self.num_items = num_items

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        entry = self.data.iloc[index]
        user_id = torch.tensor(entry["user_id"], device=self.device)
        movie_id = torch.tensor(entry["movie_id"], device=self.device)
        rating = torch.tensor(entry["rating"], device=self.device)
        return user_id, torch.sparse_coo_tensor(movie_id.unsqueeze(0), rating, size=(self.num_items,))

    @staticmethod
    def sparse_collate(batch):
        user_ids, ratings = zip(*batch)
        return user_ids, torch.stack(ratings).coalesce()

    @staticmethod
    def concat_collate(batch):
        user_ids, ratings = zip(*batch)
        return user_ids, torch.stack(ratings).coalesce()


