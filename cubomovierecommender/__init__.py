import os
import pandas as pd
import pickle
from cubomovierecommender.recommender import recommend_with_NMF, recommend_most_popular, recommend_user_CF
from cubomovierecommender.interface import get_top_match, lookup_movieId, format_imdbId
from cubomovierecommender.interface import get_imdbId, get_html_text, get_src, get_href

print('My first python package')


def test_package():
    print('welcome to cubomovierecommender')


library_path = os.path.dirname(__file__)

with open(library_path + '/models/nmf.pickle', "rb") as file:
    nmf_model = pickle.load(file)              # read in from hard-drive
r_df = pd.read_csv(
    library_path +
    '/data/ratings.csv',
    sep=',',
    index_col=[1])  # read in from hard-drive
movies = pd.read_csv(
    library_path +
    '/data/movies.csv',
    sep=',',
    index_col=[0])  # read in from hard-drive
top_df = pd.read_csv(library_path + '/data/top_df.csv', sep=',', index_col=[0])
l_df = pd.read_csv(library_path + '/data/links.csv', sep=',', index_col=0)
