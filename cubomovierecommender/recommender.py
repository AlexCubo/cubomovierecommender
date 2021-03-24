"""
Contains various recommondation implementations
all algorithms return a list of movieids
"""

import numpy as np
import pandas as pd
#from cubomovierecommender.interface import movies
from cubomovierecommender.interface import clean_and_pivot_r_df, create_user_vector
from cubomovierecommender.interface import get_user_vector_df, recommend_movies
from cubomovierecommender.interface import fill_user_vector_df, clean_and_pivot_r_df
from cubomovierecommender.interface import sort_r_df



# example output of web application
user_rating = {
    'titanic': 5,
    'terminator': 2,
    'star wars': 5
}

#with open("./models/nmf.pickle", "rb") as file:
#   nmf_model = pickle.load(file)              # read in from hard-drive

def recommend_with_NMF(user_rating, top, model):
    '''user_item_matrix is a df (n_users x n_movies)
        top is the top_tit_gen df (n_movies x 3) --. cols: ave_rating, title, genre
        user_rating is a dictionary with key a keyword for film and value the rating
        model is a trained model, in this case nmf model
        returns a list of movies IDs'''
    user_vector = create_user_vector(user_rating, top)
    #transform user_vector
    user_vector_df = get_user_vector_df(user_vector)
    user_vector_df_filled = fill_user_vector_df(user_vector_df, top)
    #predict
    Q = model.components_
    P = model.transform(user_vector_df_filled)
    prediction = np.dot(P,Q)
    recommendations = pd.DataFrame(prediction,columns=user_vector_df_filled.columns)
    recom_moviesId = recommend_movies(recommendations, user_vector_df)
    return recom_moviesId[:5]


def recommend_random(movies, user_rating, k=5):
    """
    return k random unseen movies for user 
    """
    return None


def recommend_most_popular(user_rating, top):
    ''' user_rating is a dic with key a film title (or part of it) and value a rating (0 to 5)'''
    #user_item_matrix = clean_and_pivot_r_df(r_df).T
    user_vector = create_user_vector(user_rating, top)
    user_vector_df = get_user_vector_df(user_vector)
    user_vector_filled = fill_user_vector_df(user_vector_df, top)
    recom_moviesId = recommend_movies(user_vector_filled, user_vector_df)
    return recom_moviesId[:5]


def recommend_user_CF(user_rating, r_df, top):
    r_df = clean_and_pivot_r_df(r_df)
    r_df_sorted = sort_r_df(r_df)
    user_vector = create_user_vector(user_rating, top)
    user_vector_df = get_user_vector_df(user_vector)
    last_user = list(r_df_sorted.columns)[-1]
    r_df_sorted[last_user+1] = user_vector_df
    r_df_sorted.fillna(r_df_sorted.mean(), inplace=True) #ask about np.nanmean
    UU = r_df_sorted.corr()
    active_user = list(UU.columns)[-1]
    neighboors = UU.loc[active_user]
    neighboors = neighboors.sort_values(ascending=False).iloc[1:11]
    bool_mask_df = user_vector_df.isnull()
    #bool_mask = pd.isnull(user_vector_df.T.values[0])
    neighboors_ratings = r_df_sorted.T.loc[neighboors.index, bool_mask_df['rating']]
    recom_movieIds = neighboors_ratings.mean().sort_values()
    recom_moviesId = list(recom_movieIds.index)
    return recom_moviesId[:5]

def similar_movies(movieId, movie_movie_distance_matrix):
    pass



