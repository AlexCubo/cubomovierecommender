"""
INTERFACE
- movies with shape (#number of movies, #features(title, year, genres, ...))
- user_item_matrix with shape (#number of users, #number of movies)
- top_list with shape (#number of movies, 2)
- item-item matrix with shape (#number of popular movies, #number of popular movies)
- nmf_model: trained sklearn NMF model
"""
import pandas as pd
import numpy as np
from fuzzywuzzy import process
import pickle
from bs4 import BeautifulSoup
import requests

# r_df = pd.read_csv('./data/ratings.csv', sep=',', index_col=[1]) # read in from hard-drive
# movies = pd.read_csv('./data/movies.csv', sep=',', index_col=[0]) # read in from hard-drive
#top_df = pd.read_csv('./data/top_df.csv', sep=',', index_col=[0])
#l_df = pd.read_csv('./data/links.csv', sep=',', index_col=0)


# print(movies.head())

def get_top_match(movie_title, movie_list):
    # movieId,title,genres
    match = process.extract(movie_title, movie_list, limit=3)
    return match


def create_user_vector_with_title(movie_dict, movies):
    """
    convert dict of user_ratings to a user_vector
    """
    # generate the user vector
    user_vector = pd.Series(np.nan, index=movies['title'])
    user_vector[movie_dict.keys()] = movie_dict.values()
    return user_vector


def clean_and_pivot_r_df(r_df, k=10):
    r_df = r_df.drop(columns=['timestamp'])
    r_df = r_df.pivot(columns='userId', values='rating')
    # selecting only movies that have more than 10 rates
    r_df = r_df[r_df.notna().sum(axis=1) > k]
    return r_df


def sort_r_df(r_df):
    '''r_df is dataframe, usuallu the one processed by clean_and_pivot_r_df
        returns r_df sorted by descending average rating'''
    rating_ave = np.nanmean(
        r_df.values,
        axis=1)  # row average (on the movies Id) for each user
    r_df['ave_rating'] = pd.Series(rating_ave, index=r_df.index)
    r_df_sorted = r_df.sort_values('ave_rating', ascending=False)
    r_df_sorted.drop(columns=['ave_rating'], inplace=True)
    return r_df_sorted


def get_user_vector_df(user_vector):
    '''user_vector is a dict. key:movieID, value:ratings
        Transform the user_vector in df'''
    user_vector = pd.DataFrame(
        list(
            user_vector.values()), index=list(
            user_vector.keys()))
    user_vector.columns = ['rating']
    return user_vector


def fill_user_vector_df(user_vector_df, top):
    '''user_vector_df is the df of user_vector
       Fills the df with ave_ratings'''
    user_vector_filled = user_vector_df.loc[top.index]['rating'].fillna(
        top.loc[top.index]['ave_rating'])
    user_vector_filled = pd.DataFrame(user_vector_filled).T
    return user_vector_filled


def recommend_movies(prediction_df, user_vector_df):
    '''prediction_df is a dataFrame with a prediction matrix made by a whatever model
       user_vector_df is the df version of a user_vector (result of get_user_vector_df)
       Returns a list of recommended movieIds'''
    bool_mask_df = user_vector_df.isnull()
    not_rated = prediction_df.columns[bool_mask_df['rating']]
    # movies to recommend
    movies_to_recommend = prediction_df[not_rated]
    movies_to_recommend = movies_to_recommend.T
    movies_to_recommend.columns = ['predicted_rating']
    movies_to_recommend = movies_to_recommend.sort_values(
        by='predicted_rating', ascending=False)
    recommended_moviesId = list(movies_to_recommend.index)
    return recommended_moviesId


def create_user_vector(user_rating, top_tit_gen):
    '''user_rating is the output of web application --> dict {'titanic':4,'orange':5,...}
        top_tit_gen is a dataframe with videoId as index and title, genre as columns.
        user_item_matrix is a dataframe of size (n_users x n_videos).
        The function returns a dict: key is movieId and value is rating (Nan or a int)'''
    movies_list = list(top_tit_gen.index)  # list(user_item_matrix.columns)
    empty_list = [np.nan] * len(movies_list)
    ratings_dict = dict(zip(movies_list, empty_list))
    for key, value in user_rating.items():
        #title, similarity_score, movie_id = process.extract()
        res = process.extract(key, top_tit_gen['title'], limit=1)
        for r in res:
            ratings_dict[r[2]] = value
    return ratings_dict


def lookup_movieId(top, movieId):
    '''top is the top_tit_gen df (n_movies x 3) --. cols: ave_rating, title, genre
       movieId is a movie Id,
       Returns the title of movie associated to movieId'''
    title = list(top.loc[top.index == movieId]['title'])[0]
    return title


def format_imdbId(imdbId):
    '''imdbId is a int. Returns a str'''
    imdbId = str(imdbId)
    len_id = len(imdbId)
    diff = 7 - len_id
    for i in range(diff):
        imdbId = '0' + imdbId
    return imdbId


def get_imdbId(l_df, movie_id):
    return l_df.loc[l_df.index == movie_id]['imdbId'].values[0]


def get_html_text(url):
    resp = requests.get(url)
    html = resp.text
    soup = BeautifulSoup(html, features='html.parser')
    return soup


def get_src(soup):
    tag = soup.find('img')
    src = tag.get('src')
    return src


def get_href(soup):
    tag = soup.find(
        "a", {
            "class": "slate_button prevent-ad-overlay video-modal"})
    href = tag.get('href')
    return href
