import pandas as pd
import os
import urllib.request
import zipfile
import sys

def load_data():
    if not os.path.exists(f'ml-100k.zip'):
        urllib.request.urlretrieve(f'https://files.grouplens.org/datasets/movielens/ml-100k.zip', f'ml-100k.zip')

    with zipfile.ZipFile(f"ml-100k.zip", "r") as f:
        f.extractall("./")


    data = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    item_dsc = pd.read_csv('ml-100k/u.item', sep='|', names=['movie_id', 'title', 'release_date', 'video_release', 'IMDB_url', \
                                                            'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', \
                                                            'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', \
                                                             'Musical', 'Mystery', 'Romannce', 'Sci-Fi', 'Thriller', 'War', 'Western'],
                                                                encoding='iso-8859-1')

    
    # A = data.pivot(index='user_id', columns='movie_id', values='rating')
    # A = A.to_numpy()

    num_users = data['user_id'].nunique()
    num_movies = data['movie_id'].nunique()

    return data, num_users, num_movies


def train_test_split(data, test_size = 0.2):
    #movies index starts at 1
    # LOO_Movies = {i: rows.last_valid_index() -1 for i, _, rows in enumerate(A.iterrows())}
    
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



