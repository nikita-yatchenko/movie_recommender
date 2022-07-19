# Movie Recommender System

The goal of this project is to build a web application
 that would recommend movies to a user utilizing various machine learning approaches.

## <u> Quick recommender system overview </u>
Recommender system is a subclass of  machine learning information filtering 
models that seek to predict users' ratings of individual items based on either 
content or collaborative filterings.
- Content filtering - model uses items' set of characteristics to 
find similarity
- Collaborative filtering - model that uses past user behavior and other user's behavior 
to pull together an estimate of what a current user might like

### Challenges



## <u> Movie Recommender Project </u> 
### Dataset
We are going to be using a popular movie dataset [MovieLens](https://grouplens.org/datasets/movielens/latest/)


To run make a single recommendation:
0) install necessary dependencies `pip install -r requirements.txt`
1) go to ../movie_recommender/src/rec_sys
2) run `python -W ignore .\knn_recommender.py --path '[path]\data' --movie_name '[movie name]' --top_n [n]`