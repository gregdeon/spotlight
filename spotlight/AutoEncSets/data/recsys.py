import pandas as pd
import numpy as np
import os

# from data import CompletionDataset
from . import CompletionDataset

def ml100k(validation=0., seed=None, path='./data/ml-100k', return_test=False):
    rng = np.random.RandomState(seed)
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    train_df = pd.read_csv(path + "/u1.base", sep="\t", names=r_cols, encoding='latin-1')
    validation_df = pd.read_csv(path + "/u1.test", sep="\t", names=r_cols, encoding='latin-1')
    
    ratings = np.concatenate([train_df.rating, validation_df.rating])
    mask = np.concatenate([np.array([train_df.user_id, train_df.movie_id]).T,
                           np.array([validation_df.user_id, validation_df.movie_id]).T], axis=0)
     
    if validation > 0.:
        n = train_df.shape[0]
        n_train = int(n * (1-validation))
        ind_tr = rng.permutation(np.concatenate([np.zeros(n_train), np.ones(n - n_train)]))
    else:
        ind_tr = np.zeros_like(train_df.rating)
        
    indicator = np.concatenate([ind_tr, 2 * np.ones_like(validation_df.rating)])
    return CompletionDataset(ratings, mask, indicator, return_test=return_test)


def ml1m(validation=0., test=0.1, seed=None):
    rng = np.random.RandomState(seed)
    r_cols = ['user_id', None, 'movie_id', None, 'rating', None, 'unix_timestamp']

    ratings_df = pd.read_csv('ml-1m/ratings.dat', sep=':', names=r_cols, encoding='latin-1')
    
    n_ratings = ratings_df.rating.shape[0]
    n_users = np.max(ratings_df.user_id)

    _, movies = np.unique(ratings_df.movie_id, return_inverse=True)
    n_movies = np.max(movies) + 1

    n_ratings_val = int(n_ratings * validation)
    n_ratings_ts = int(n_ratings * test)
    n_ratings_tr = n_ratings - n_ratings_val - n_ratings_ts

    indicator = np.concatenate((np.zeros(n_ratings_tr, np.int32), 
                                   np.ones(n_ratings_val, np.int32), 
                                   2 * np.ones(n_ratings_ts, np.int32)))
    indicator = rng.permutation(indicator)
    ratings = ratings_df.rating
    mask = np.array(list(zip(ratings_df.user_id-1, ratings_df.movie_id)))
    return CompletionDataset(ratings, mask, indicator)


