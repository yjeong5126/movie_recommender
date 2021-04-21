# Movie Recommendation Engines
> This is the practice of making movie recommendation engines. 

## Goal of This Project
The goal of this project is to understand how each model of the recommendation engines works and practice making the recommenders with movie data sets.

## Content-Based Filtering

<img src="https://github.com/yjeong5126/movie_recommender/blob/master/images/tfidf1.PNG" title="Feature1" width="600" height="250">

**Content-based recommender** is the system to rely on the similarity of items when it recommends items to users. For example, when a user likes a movie, the system finds and recommends movies which have more similar features to the movie the user likes.
In the feature above, *Movie 1* and *Movie 2* are considered similar each other and they are not similar to *Movie 3*. If a user likes *Movie 1*, then the system should recommend *Movie 2* to the user.

### How to Implement
- Step 1: Quantify the features of each movie in the data set (Use **Term Frequency and Inverse Document Frequency (tf-idf)**)
- Step 2: Calculate the similarity between movies (Use **Cosine Similarity**)
- Step 3: Build the recommendation algorithm

#### Term Frequency and Inverse Document Frequency (tf-idf)
**tf-idf** is a numerical statistic which is used to calculate the importance of a word to a document in a collection of documents. The basic formula is as follows:

*tf-idf (i, j) = tf (i, j) × idf (i, N)*

- tf (i, j) = f (i, j) / ∑ₖ f(k, j)
- idf (i, N) = log(N/df (i))
- f (i, j): the number of times that word i occurs in document j
- ∑ₖ f(k, j) : the number of words in document j
- df ᵢ: the number of documents where the word i appears
- N: the total number of documents.

The **TfidVectorizer()** class from **sklearn.feature_extraction.text** library can be used to calculate and vectorize the tf-idf scores for each movie.

#### Cosine Similarity
we can use the cosine similarity which can be used to calculate the distance between two vectors. The formula of the cosine similarity is as follows:

*Similarity = cos(θ) = (A⋅B)/(∥A∥×∥B∥)*

- A & B: non-zero vectors
- θ: the measure of the angle between A and B
- A⋅B: dot product
- ∥A∥ or ∥B∥: the length of the vector A or B

The **linear_kernel()** class in **sklearn.metrics.pariwise** can be used to calculate the cosine similarity.

#### 'Did you mean...?' Trick
We often misspell a movie title. When we make misspellings while using *Google* to search something, *Google* asks us, **‘Did you mean…?’** in order to help our search. 
I apply *Levenshtein Distance* in order to implement this trick to the recommendation engine. This is a technique to calculate the distance between words. 
The **fuzz** class in **fuzzywuzzy** library can be used to implement the *Levenshtein Distance* in Python.

### Data 
For this practice, I use a movie dataset from the **MovieLens**. The version used in this practice has 9,742 movies. ([movies.csv](https://github.com/yjeong5126/movie_recommender/blob/master/content_based/movies.csv))

<img src="https://github.com/yjeong5126/movie_recommender/blob/master/images/movies_head.PNG" width="700" height="200">

### Build the Content-based Movie Recommender
All the lines of the code to build the content-based movie recommender are [here](https://github.com/yjeong5126/movie_recommender/blob/master/content_based/content_based_recommender.ipynb).
The additional explanation about the logic and the code can be found in this [page](https://yjeong5126.medium.com/creating-content-based-movie-recommender-with-python-7f7d1b739c63) as well.

## Item-based Collaborative Filtering

**Item-based collaborative filtering** is the recommendation system to use the similarity between items using the ratings by users. The fundamental assumption for this method is that a user gives similar ratings to similar movies.

<img src="https://github.com/yjeong5126/movie_recommender/blob/master/images/rating_sample.PNG" title="Feature1" width="600" height="180">

If we predict the rating for **Movie_1** by **User_1** in the example above:

- Step 1: Find the most similar (the nearest) movies to the movie for which you want to predict the rating.
- Step 2: Calculate the weighted average of the ratings for the most similar movies by the user.

The way to find the similar movies is to use the **cosine similarity** as follows:

<img src="https://github.com/yjeong5126/movie_recommender/blob/master/images/cosine_similarity.PNG" title="Feature1" width="500" height="250">

In using the cosine similarity, replace the missing value for 0. **Movie_3** and **Movie_0** are the nearest movies to **Movie_1**. Then, the weighted average of the ratings for **Movie_3** and **Movie_0** by **User_1** will be the predicted rating for **Movie_1** by **User_1**. The cosine similarity is used for the weight for each similar movie.  

<img src="https://github.com/yjeong5126/movie_recommender/blob/master/images/rating_sample2.PNG" title="Feature1" width="600" height="200">

### Data 
For this practice, I create a small size of a rating dataset. Then, I apply the code I create to the **MovieLens** dataset. ([movies.csv](https://github.com/yjeong5126/movie_recommender/blob/master/item_based_collaborative_filtering/movies.csv), [ratings.csv](https://github.com/yjeong5126/movie_recommender/blob/master/item_based_collaborative_filtering/ratings.csv))

### Build the Item-based Collaborative Filtering
All the lines of the code to build the item-based collaborative filtering are [here](https://github.com/yjeong5126/movie_recommender/blob/master/item_based_collaborative_filtering/item_based_collaborative_filtering.ipynb).
The additional explanation about the logic and the code can be found in this [page](https://yjeong5126.medium.com/item-based-collaborative-filtering-in-python-91f747200fab) as well.


