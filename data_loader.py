import pandas as pd
import os
import urllib.request
import zipfile
from torch.utils.data import Dataset, DataLoader

# from tqdm.notebook import tqdm
from tqdm import tqdm

import numpy as np

def load_data(size, rescale_data=False, negative_samples=20):
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


    unique_users = data['user_id'].unique()
    print("Number of users", len(unique_users))
    unique_movies = data['movie_id'].unique()
    print("Number of movies", len(unique_movies))
    
    if rescale_data:
        # IMPORTANT: For regression tasks we need to rescale the ratings between 0,1
        data['rating'] = data['rating'] / 5.0
    else:
        data['rating'] = 1.0
        new_data = data.copy()
        new_rows = []
        for user in tqdm(unique_users):
            user_data = data.loc[data['user_id'] == user]
            movies_for_user = user_data['movie_id'].unique()

            movie_samples = np.random.choice(np.setdiff1d(unique_movies, movies_for_user), negative_samples)

            new_rows.extend([{'user_id': user,'movie_id': sample, 'rating': 0} for sample in movie_samples])
    
        new_data_frame = pd.DataFrame(new_rows)
        new_data = pd.concat([new_data, new_data_frame], ignore_index=True)
        data = new_data

    ##zero index for embedding layer
    user_map = {old : new for new, old in enumerate(unique_users)}
    data['user_id'] = data.user_id.map(user_map)
    movie_map = {old : new for new, old in enumerate(unique_movies)}
    data['movie_id'] = data.movie_id.map(movie_map)

    ##user item size
    num_users = unique_users.shape[0]
    num_movies = unique_movies.shape[0]
    
    # sort by user_id to ensure things are grouped naturally as before potentially adding rows
    data = data.sort_values('user_id').reset_index()

    return data, num_users, num_movies


def train_test_split(data, test_size = 0.3):
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



class MovielensDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        return self.users.shape[0]
    
    def __getitem__(self, index):
        return self.users[index], self.movies[index], self.ratings[index]