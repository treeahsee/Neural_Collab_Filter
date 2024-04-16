import pandas as pd
import os
import urllib.request
import zipfile
from torch.utils.data import Dataset, DataLoader

def load_data(size):
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



class MovielensDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        return self.users.shape[0]
    
    def __getitem__(self, index):
        return self.users[index], self.movies[index], self.ratings[index]