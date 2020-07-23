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

N = ratings['userId'].nunique()
M = ratings['movieId'].nunique()

user_mapper = dict(zip(np.unique(ratings['userId']), list(range(N))))
movie_mapper = dict(zip(np.unique(ratings['movieId']), list(range(M))))
    
user_inv_mapper = dict(zip(list(range(N)), np.unique(ratings['userId'])))
movie_inv_mapper = dict(zip(list(range(M)), np.unique(ratings['movieId'])))

def load_sparse_csr(filename):
    # here we need to add .npz extension manually
    loader = np.load(filename + '.npz')
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

X = load_sparse_csr('resources/models/sparse_matrix')

def find_similar_movies(movie_id, k, metric='cosine', show_distance=False):
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
    similar_ids = find_similar_movies(movie_id, k=10)
    movie_title = movie_titles[movie_id]
    y = []
    for i in similar_ids:
        y.append(movie_titles[i])
    return y

def extract_year_from_title(title):
    t = title.split('(')
    year = None
    if re.search(r'\d+\)', t[-1]):
        year = t[-1].strip(')')
        year = year.replace(')', ' ')
    return year

movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))
movies['year'] = movies['title'].apply(extract_year_from_title)
movies = movies[~movies['year'].isnull()]

rating = ratings.merge(movies, on ='movieId')
rating.dropna(inplace=True)

def top_rated(years, genre):
    df = rating[rating['year'] == years]
    comedy_movies = []
    
    for row,col in df.iterrows():
        if genre in col['genres']:
            if col['title'] not in comedy_movies:
                comedy_movies.append(col['title'])
    
    comedy_m = {}
    for i in comedy_movies:
        comedy_m[i] = df[df['title']==i]['rating'].sum()/ len(df[df['title'] == i]['rating'])
    
    c_mm = {k: v for k, v in sorted(comedy_m.items(), key=lambda item: item[1])}
    g = list(c_mm.keys())
    y = (g[::-1])[:10]
    
    return "\n".join(y)

genres_counts = Counter(g for genres in movies['genres'] for g in genres)
del genres_counts['(no genres listed)']

def genres():
    return list(genres_counts.keys())