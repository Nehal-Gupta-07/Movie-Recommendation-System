import pandas as pd
from helper_functions import *

movies = pd.read_csv(r"../data/tmdb_5000_movies.csv")
credits = pd.read_csv(r"../data/tmdb_5000_credits.csv")

movies = movies.merge(credits,on='title')

def preprocess(movies):

    movies = movies[['id','genres','keywords','title','cast','crew','overview']]

    movies.dropna(inplace=True)

    movies['cast'] = movies['cast'].apply(convert5)
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x:x.split())

    movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
    movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
    movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew'] 

    new_df = movies[['id','title','tags']]

    new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))

    new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

    return new_df

new_df = preprocess(movies)
new_df.to_csv(r"../data/preprocessed_data", encoding='utf-8')
