"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re

# Importing data
movies = pd.read_csv('resources/data/movies.csv', sep = ',',delimiter=',')
ratings = pd.read_csv('resources/data/ratings.csv')
movies.dropna(inplace=True)

#Preprocessing on the movies dataframe
movies = movies[:50000]
movies['genres'] = movies['genres'].apply(lambda x: x.split("|"))
genres_counts = Counter(g for genres in movies['genres'] for g in genres)
del genres_counts['(no genres listed)']

def extract_year_from_title(title):
    t = title.split('(')
    year = None
    if re.search(r'\d+\)', t[-1]):
        year = t[-1].strip(')')
        year = year.replace(')', ' ')
    return year

movies['year'] = movies['title'].apply(extract_year_from_title)
movies = movies[~movies['year'].isnull()]

def get_decade(year):
    year = str(year)
    decade_prefix = year[0:3] # get first 3 digits of year
    decade = f'{decade_prefix}0' # append 0 at the end
    return int(decade)

movies['decade'] = movies['year'].apply(get_decade)
genres = list(genres_counts.keys())

for g in genres:
    movies[g] = movies['genres'].transform(lambda x: int(g in x))
    
movie_decades = pd.get_dummies(movies['decade'])
movie_idx = dict(zip(movies['title'], list(movies.index)))
movie_features = pd.concat([movies[genres], movie_decades], axis=1)

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def content_model(movie_list,top_n=10):
    """Performs Content filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """   
    # Generating the cosine similarity matrix
    cosine_sim = cosine_similarity(movie_features, movie_features)
    
    # Getting the index of the movie that matches the title
    idx_1 = movie_idx[movie_list[0]]
    idx_2 = movie_idx[movie_list[1]]
    idx_3 = movie_idx[movie_list[2]]
     
    # Calculating the similarity scores
    sim_score_1 = cosine_sim[idx_1]
    sim_score_2 = cosine_sim[idx_2]
    sim_score_3 = cosine_sim[idx_3]
    
    # Creating a Series with the similarity scores in descending order
    sorted_score_1 = pd.Series(sim_score_1).sort_values(ascending = False)
    sorted_score_2 = pd.Series(sim_score_2).sort_values(ascending = False)
    sorted_score_3 = pd.Series(sim_score_3).sort_values(ascending = False)
    
    # Appending all the scores to a single Series in descending order  
    scores = sorted_score_1.append(sorted_score_2).append(sorted_score_3).sort_values(ascending = False)

    # Generating empty list to store movie names
    recommendations = []
    
    # Getting the indexes of the 50 most similar movies
    top_50_indexes = scores.iloc[1:50].index.tolist()
    
    # Removing the chosen movies
    top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
    
    # Appending Top 10 Recommendations to empty list
    for i in top_indexes[:top_n]:
        recommendations.append(movies['title'][i])
        
    return recommendations