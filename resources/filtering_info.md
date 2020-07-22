## Collaborative-based Filtering

**Collaborative filtering** is based on the concept of "homophily" - similar users like similar things. It uses movie preferences from other users to predict which movie a particular user will like best. Collaborative filtering uses a user-item matrix to generate recommendations. This matrix is populated with values that indicate a given user's preference towards a given movie. It's very unlikely that a user will have interacted with every movie, so in most real-life cases, the user-item matrix is very sparse.

![user-item matrix](https://raw.githubusercontent.com/Unsupervised-TeamSS3/unsupervised-predict-streamlit-template/master/resources/imgs/user_item_matrix.png)

A major disadvantage of collaborative filtering is the **cold start problem**. You can only get recommendations for users and movies that already have "interactions" in the user-item matrix. Collaborative filtering fails to provide personalized recommendations for brand new users or newly released movies.

## Content-based Filtering

**Content-based filtering** generates recommendations based on user and movie features. Given a set of movie features (movie genre, release date, country, language, etc.), it predicts how a user will rate a movie based on their ratings of previous movies.

Content-based filtering handles the **cold start problem** because it is able to provide personalized recommendations for brand new users and features.


![content](https://raw.githubusercontent.com/Unsupervised-TeamSS3/unsupervised-predict-streamlit-template/master/resources/imgs/content.png)

