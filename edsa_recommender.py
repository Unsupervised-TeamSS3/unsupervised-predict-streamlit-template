"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model
from recommenders.memory_based_recommender import get_content_based_recommendations
from recommenders.memory_based_recommender import movie_finder
from recommenders.memory_based_recommender import top_rated
from recommenders.memory_based_recommender import genres

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')                    
                     
# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["About the app", "Content vs Collaborative", "Data Insights", "Recommender System", "Because You Watched ...", "Blockbusters"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('First Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "About the app":
        
        html_temp = """
        <div style="background-color:black;padding:10px">
        <h1 style="color:red;text-align:center;">Movie Recommender App </h2>
        </div>
        """
        st.markdown(html_temp,unsafe_allow_html=True)
        st.info("General Information about the app")
        
        st.markdown(open('resources/info.md').read())
    
    # ------------------------------------------------------------------
                     
    if page_selection == "Data Insights":
        
        html_temp = """
        <div style="background-color:black;padding:10px">
        <h1 style="color:red;text-align:center;">Movie Recommender App </h2>
        </div>
        """
        st.markdown(html_temp,unsafe_allow_html=True)
        st.info("Insights on the movies")
                     
        st.markdown("The following page contains visuals related to the movies dataset.")             
        
        st.subheader("View visuals:")
        
        if st.checkbox('Distribution of Genres'):
            st.image('resources/imgs/Genre.PNG',use_column_width=True)         
                     
        if st.checkbox('Decade of Release of Movies'):
            st.image('resources/imgs/Decades.PNG',use_column_width=True)          
                     
        if st.checkbox('Distribution of Ratings'):
            st.image('resources/imgs/Ratings.PNG',use_column_width=True)          
                     
        if st.checkbox('Highest Rated Movies (Bayesian Average)'):              
            st.image('resources/imgs/Highest_Rated.PNG',use_column_width=True)          
                     
        if st.checkbox('Lowest Rated Movies (Bayesian Average)'):              
            st.image('resources/imgs/Lowest_Rated.PNG',use_column_width=True)                              
                     
    # ------------------------------------------------------------------
                     
    if page_selection == "Because You Watched ...":
        
        html_temp = """
        <div style="background-color:black;padding:10px">
        <h1 style="color:red;text-align:center;">Movie Recommender App </h2>
        </div>
        """
        st.markdown(html_temp,unsafe_allow_html=True)
        st.info("Movie Recommender")
        
        movie_text = st.text_area("Enter your favourite movie:", "Type Here")
        
        if st.button("Get 10 Recommendations"):
            try:
                with st.spinner('Crunching the numbers...'):
                    reco = get_content_based_recommendations(movie_text)
                    string = '\n'.join(reco)
                    movie_name = movie_finder(movie_text)
                    st.success(f'### BECAUSE YOU WATCHED {movie_name.upper()}:')
                    st.text(string)
            except:
                    st.error("Oops! Looks like we cannot find your movie!")
            
    # -------------------------------------------------------------------
    
    if page_selection == "Content vs Collaborative":
        
        html_temp = """
        <div style="background-color:black;padding:10px">
        <h1 style="color:red;text-align:center;">Movie Recommender App </h2>
        </div>
        """
        st.markdown(html_temp,unsafe_allow_html=True)
        st.info("Explanation on Content-based filtering and Collaborative-based filtering")
        
        st.markdown(open('resources/filtering_info.md').read())
        
   # --------------------------------------------------------------------

    if page_selection == "Blockbusters":
        
        html_temp = """
        <div style="background-color:black;padding:10px">
        <h1 style="color:red;text-align:center;">Movie Recommender App </h2>
        </div>
        """
        st.markdown(html_temp,unsafe_allow_html=True)
        st.info("Getting the best 10 movies by year of release and genre")
        
        z = list(range(1980, 2020))
        string = [str(integer) for integer in z]
        
        st.write('### Select the genre')
        
        gen = st.selectbox('Genre',genres())
        
        st.write('### Select the year')
        
        yen = st.selectbox('Year', string)
        
        #reco = top_rated(yen, gen)
        
        if st.button("Get Movies"):
            with st.spinner('Crunching the numbers...'):
                reco = top_rated(yen, gen)
                if reco != "":
                    st.success(f'### THE BEST {gen.upper()} MOVIES OF {yen} ARE:')
                    st.text(reco)
                else:
                    st.error("Oops! Looks like there aren't any movies that fit this description!")

if __name__ == '__main__':
    main()
