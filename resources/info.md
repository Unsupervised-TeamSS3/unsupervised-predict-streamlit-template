## **How to use the app**

In the **"Content vs Collaborative"** tab, the user can gain an understanding of the difference between content-based filtering and collaborative-based filtering.

In the **"Data Insights"** tab, one can have a look at:

1. The distribution of genres
2. The distribution of the decades of the release of movies
3. The distribution of the ratings
4. The highest-rated movies 
5. The lowest-rated movies 

(Note that 4. and 5. were calculated using the [Bayesian average](https://www.evanmiller.org/bayesian-average-ratings.html), which takes into account the number of ratings. For interest sake, the average and bayesian ratings can be seen).

In the **"Recommender System"** tab, the user can select 3 of their favourite movies from the available list of movies, and the app will recommend 10 movies for the user to watch. The user should choose the *content-based filtering* option for this to work. 

In the **"Because You Watched ..."** tab, the user simply types in their favourite movie (even if the spelling is wrong), and the app will recommend 10 movies for the user to watch. This feature makes use of the *collaborative-based* filtering method.

In the **"Blockbusters"** tab, the user simply chooses a year and a genre, and the app will inform the user of the best (maximum 10) movies of that year and genre, according to the ratings. 


## **The data**

This dataset consists of several million ratings obtained from users of the online MovieLens movie recommendation service. The MovieLens dataset has long been used by industry and academic researchers to improve the performance of explicitly-based recommender systems.

The data for the MovieLens dataset is maintained by the GroupLens research group in the Department of Computer Science and Engineering at the University of Minnesota. Additional movie content data was legally scraped from IMDB.