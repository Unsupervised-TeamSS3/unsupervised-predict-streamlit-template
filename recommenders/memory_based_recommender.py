"""

    Collaborative-based filtering for item recommendation.

    Author: Team SS3 JHB

    Description: Provided within this file is a baseline memory-based 
    collaborative filtering algorithm for rating predictions on Movie data.

"""   
    
# Script dependencies
import pandas as pd
import numpy as np

#modelling libraries
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

#other libraries
import re 
from collections import Counter
from fuzzywuzzy import process

# Importing data
movies = pd.read_csv('resources/data/movies.csv',sep = ',',delimiter=',')
ratings = pd.read_csv('resources/data/ratings.csv')
ratings.drop(['timestamp'], axis=1,inplace=True)   
    
def create_X(df):
    """
    Generates a sparse matrix from ratings dataframe.
    
    Args:
        df: pandas dataframe
    
    Returns:
        X: sparse matrix
        user_mapper: dict that maps user id's to user indices
        user_inv_mapper: dict that maps user indices to user id's
        movie_mapper: dict that maps movie id's to movie indices
        movie_inv_mapper: dict that maps movie indices to movie id's
    """
    
    N = df['userId'].nunique()
    M = df['movieId'].nunique()

    user_mapper = dict(zip(np.unique(df['userId']), list(range(N))))
    movie_mapper = dict(zip(np.unique(df['movieId']), list(range(M))))
    
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df['userId'])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df['movieId'])))
    
    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df['rating'], (movie_index, user_index)), shape=(M, N))
    
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_X(ratings)

def find_similar_movies(movie_id, X, k, metric='cosine', show_distance=False):
    """
    Finds k-nearest neighbours for a given movie id.
    
    Args:
        movie_id: id of the movie of interest
        X: user-item utility matrix
        k: number of similar movies to retrieve
        metric: distance metric for kNN calculations
    
    Returns:
        list of k similar movie ID's
    """
    neighbour_ids = []
    
    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    k+=1
    kNN = NearestNeighbors(n_neighbors=k, algorithm='brute', metric=metric)
    kNN.fit(X)
    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1,-1)
    neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)
    for i in range(0,k):
        n = neighbour.item(i)
        neighbour_ids.append(movie_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids

def movie_finder(title):
    all_titles = movies['title'].tolist()
    closest_match = process.extractOne(title,all_titles)
    return closest_match[0]

movie_idx = dict(zip(movies['title'], movies['movieId']))
movie_titles = dict(zip(movies['movieId'], movies['title']))

def get_content_based_recommendations(title_string):
    title = movie_finder(title_string)
    idx = movie_idx[title]
    movie_id = idx
    similar_ids = find_similar_movies(movie_id, X, k=10)
    movie_title = movie_titles[movie_id]
    y = []
    for i in similar_ids:
        y.append(movie_titles[i])
    return y