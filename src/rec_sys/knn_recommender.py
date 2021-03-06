import pandas as pd
import numpy as np

# Data Science
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

# utils
import os
import time
import gc
import argparse
from fuzzywuzzy import fuzz


class KNNRecommender:
    '''
    Recommender class based on the KNN algorithm with collaborative filtering

    Input:
        - path_movies (path to movie data)
        - path_ratings (path to ratings data)
    '''
    def __init__(self, path_movies, path_ratings, user_threshold,
                 movie_threshold, svd_features=10, normalize=False):
        '''
        Setting up class instance: need path to movies and ratings
        Setting up the thresholds to get rid of underreviewed movies and inactive users

        :param path_movies
        :param path_ratings
        :param user_threshold: minimum number of movies a user would have to have scored
        :param movie_threshold: minimum number of users a movie would have to have reviewed
        '''
        self.path_movies = path_movies
        self.path_ratings = path_ratings
        self.user_threshold = user_threshold
        self.movie_threshold = movie_threshold
        self.svd_features = svd_features
        self.normalize = normalize

        print('SVD features: ', self.svd_features)
        print('Normalize: ', self.normalize)

        self.movies = pd.read_csv(self.path_movies,
                                  usecols=['movieId', 'title'],
                                  dtype={'movieId': 'int32', 'title': 'str'})
        self.ratings = pd.read_csv(self.path_ratings,
                                   usecols=['userId', 'movieId', 'rating'],
                                   dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

        self.movie_user, self.movie2ind_dict, self.ind2movie_dict = self._data_prep(self.movies,
                                                                                    self.ratings,
                                                                                    self.user_threshold,
                                                                                    self.movie_threshold,
                                                                                    self.normalize)
        self.movie_features = self._get_movie_features()
        self.model = NearestNeighbors()

    @staticmethod
    def _data_prep(ratings, movies, user_threshold, movie_threshold, normalize):
        '''
        Merges two datasets into movies by ratings database,
        filters through the database to get rid of underrepresented movies and inactive users,
        sets up the movie to index and index to movie dictionaries

        :return:
            - proc_movie_data (filtered movie by user ratings sparse matrix)
            - movie2ind_dict
            - ind2movie_dict
        '''
        merged = pd.merge(ratings, movies, how='inner', on='movieId')
        merged = merged.groupby(by=['userId', 'title'], as_index=False).agg({"rating": "mean"})

        movie_votings = merged.groupby('title')['rating'].agg('count')
        user_votings = merged.groupby('userId')['rating'].agg('count')

        movies2use = movie_votings[movie_votings >= movie_threshold].index.values
        users2use = user_votings[user_votings >= user_threshold].index.values

        proc_movie_data = merged[merged.title.isin(movies2use) & merged.userId.isin(users2use)]

        movie_user = proc_movie_data.pivot(index='title', columns='userId', values='rating')
        if normalize:
            mask = np.isnan(movie_user)
            masked_arr = np.ma.masked_array(movie_user, mask)
            temp_mask = masked_arr.T
            rating_means = np.mean(temp_mask, axis=0)

            filled_matrix = temp_mask.filled(rating_means)
            filled_matrix = filled_matrix.T
            filled_matrix = filled_matrix - rating_means.data.reshape(-1, 1)
        else:
            filled_matrix = movie_user.fillna(0)

        movie2ind_dict = {movie: i for i, movie in enumerate(movie_user.index)}
        ind2movie_dict = {v: k for k, v in movie2ind_dict.items()}

        # clean up
        del proc_movie_data, users2use, movies2use
        del movie_votings, user_votings
        gc.collect()

        return filled_matrix, movie2ind_dict, ind2movie_dict

    def _get_movie_features(self):
        '''
        Applies TruncatedSVD to obtain a matrix of movie features

        :return:
            - movie_features
        '''
        movie_svd = TruncatedSVD(n_components=self.svd_features, random_state=42)
        filled_matrix = movie_svd.fit_transform(self.movie_user)
        print(filled_matrix.shape)
        return filled_matrix

    def set_model_params(self, n_neighbors, algo, metric, njobs):
        '''
        Setting model parameters to a custom ones

        :param n_neighbors:
        :param algo:
        :param metric:
        :param njobs:
        '''
        self.model.set_params(**{
            'n_neighbors': n_neighbors,
            'algorithm': algo,
            'metric': metric,
            'n_jobs': njobs})

    def _fuzzy_matching(self, fav_movie):
        """
        Returns the closest match via fuzzy ratio.
        If no match found, return None

        :param
            - movie2ind: dict, map movie title name to index of the movie in data
            - fav_movie: str, name of user input movie
        :return:
            - index of the closest match
        """
        match_tuple = []
        # get match
        for title, idx in self.movie2ind_dict.items():
            ratio = fuzz.ratio(title.lower(), fav_movie.lower())
            if ratio >= 60:
                match_tuple.append((title, idx, ratio))
        # sort
        match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
        if not match_tuple:
            raise ValueError('No match is found')
        else:
            print('Found possible matches in our database: '
                  '{0}\n'.format([x[0] for x in match_tuple]))
            return match_tuple[0][1]

    @staticmethod
    def _distance(movie_vector, total_movie_features, model, top_n):
        '''
        Calculates distance between a vector and the rest of the data
        :param
            - movie_vector - vectorized representation of a movie in the dataset
            - total_movie_features - vectorized representations of all movies
        :return:
            - top_n smallest distances between fav movie and rest
        '''
        model.fit(total_movie_features)
        distances, indices = model.kneighbors(
            movie_vector,
            n_neighbors=top_n+1
        )
        # print(distances, indices)
        # print(distances.shape, indices.shape)
        sort_dist_indx = sorted(
            list(
                zip(distances.flatten(), indices.flatten())
            ),
            key=lambda x: x[0]
        )
        return sort_dist_indx

    def make_recommendations(self, movie_name, top_n):
        '''
        Given a movie name - return a list of top n recommendations (movies with highest cosine similarity
        measure)

        :param movie_name: name of a movie
        :param top_n: top n recommendations
        :return: top_recs: top movie recommendations
        '''
        t0 = time.time()
        top_recs = []
        closest_movie_idx = self._fuzzy_matching(movie_name)
        movie_vec = self.movie_features[closest_movie_idx, :].reshape(1,-1)
        # print(movie_vec)
        # print(closest_movie_idx)
        sort_dist_indx = self._distance(movie_vec, self.movie_features, self.model, top_n)
        for i, val in enumerate(sort_dist_indx[1:]):
            dist = val[0]
            idx = val[1]
            print('{0}: {1}, with distance '
                  'of {2}'.format(i + 1, self.ind2movie_dict[idx], dist))
            top_recs.append(self.ind2movie_dict[idx])
        t1 = time.time()
        print('It took {:.2f}s to make inference \n\
                      '.format(t1 - t0))
        return top_recs


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Movie Recommender",
        description="Run KNN Movie Recommender")
    parser.add_argument('--path', nargs='?', default=r'../data',
                        help='input data path')
    parser.add_argument('--movies_filename', nargs='?', default='movies.csv',
                        help='provide movies filename')
    parser.add_argument('--ratings_filename', nargs='?', default='ratings.csv',
                        help='provide ratings filename')
    parser.add_argument('--movie_name', nargs='?', default='',
                        help='provide your favourite movie name')
    parser.add_argument('--top_n', type=int, default=5,
                        help='top n movie recommendations')
    parser.add_argument('--svd_features', type=int, default=10,
                        help='top n movie recommendations')
    return parser.parse_args()


if __name__ == '__main__':
    # get args
    args = parse_args()
    data_path = args.path
    movies_filename = args.movies_filename
    ratings_filename = args.ratings_filename
    movie_name = args.movie_name
    top_n = args.top_n
    svd_features = args.svd_features
    # filter params
    user_threshold=5.
    movie_threshold=4.
    normalize=False
    # initial recommender system
    recommender = KNNRecommender(
        os.path.join(data_path, movies_filename),
        os.path.join(data_path, ratings_filename),
        user_threshold,
        movie_threshold,
        svd_features,
        normalize
    )
    # set model params
    algo = 'brute'
    metric = 'euclidean'
    njobs = -1
    recommender.set_model_params(top_n, algo, metric, njobs)

    # make recommendations
    recommender.make_recommendations(movie_name, top_n)

