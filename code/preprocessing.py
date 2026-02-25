import pandas as pd
from helper_functions import *

movies = pd.read_csv(r"../data/tmdb_5000_movies.csv")
credits = pd.read_csv(r"../data/tmdb_5000_credits.csv")

movies = movies.merge(credits,on='title')

def preprocess(movies):

    movies = movies[['id','genres','keywords','title','cast','crew','overview',
                      'vote_average','release_date']]

    movies.dropna(subset=['id','title','overview','genres','keywords','cast','crew'], inplace=True)

    movies['overview_text'] = movies['overview']

    movies['cast'] = movies['cast'].apply(convert5)
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x:x.split())

    movies['genres_display'] = movies['genres'].apply(lambda x:", ".join(x))

    movies['text_genres'] = movies['genres'].apply(lambda x:" ".join([i.replace(" ","") for i in x]).lower())
    movies['text_keywords'] = movies['keywords'].apply(lambda x:" ".join([i.replace(" ","") for i in x]).lower())
    movies['text_cast'] = movies['cast'].apply(lambda x:" ".join([i.replace(" ","") for i in x]).lower())
    movies['text_crew'] = movies['crew'].apply(lambda x:" ".join([i.replace(" ","") for i in x]).lower())
    movies['text_overview'] = movies['overview_text'].apply(lambda x:x.lower())

    movies['year'] = pd.to_datetime(movies['release_date'], errors='coerce').dt.year
    movies['year'] = movies['year'].fillna(0).astype(int)

    new_df = movies[['id','title','genres_display','year','vote_average','overview_text',
                      'text_overview','text_genres','text_keywords','text_cast','text_crew']].copy()

    return new_df

new_df = preprocess(movies)
new_df.to_csv(r"../data/preprocessed_data", index=False, encoding='utf-8')
