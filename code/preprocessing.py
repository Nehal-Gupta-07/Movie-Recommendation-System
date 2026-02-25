import os
import pandas as pd
from helper_functions import extract_names, extract_top_names, extract_director

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def load_raw_data():
    movies = pd.read_csv(os.path.join(DATA_DIR, 'tmdb_5000_movies.csv'))
    credits = pd.read_csv(os.path.join(DATA_DIR, 'tmdb_5000_credits.csv'))
    return movies.merge(credits, on='title')


def strip_spaces(names):
    return " ".join(name.replace(" ", "") for name in names).lower()


def preprocess(movies):
    movies = movies[['id', 'genres', 'keywords', 'title', 'cast', 'crew',
                      'overview', 'vote_average', 'release_date']].copy()

    movies.dropna(subset=['id', 'title', 'overview', 'genres', 'keywords',
                          'cast', 'crew'], inplace=True)

    movies['overview_text'] = movies['overview']
    movies['cast'] = movies['cast'].apply(extract_top_names)
    movies['genres'] = movies['genres'].apply(extract_names)
    movies['keywords'] = movies['keywords'].apply(extract_names)
    movies['crew'] = movies['crew'].apply(extract_director)

    movies['genres_display'] = movies['genres'].apply(lambda x: ", ".join(x))

    movies['text_overview'] = movies['overview_text'].str.lower()
    movies['text_genres'] = movies['genres'].apply(strip_spaces)
    movies['text_keywords'] = movies['keywords'].apply(strip_spaces)
    movies['text_cast'] = movies['cast'].apply(strip_spaces)
    movies['text_crew'] = movies['crew'].apply(strip_spaces)

    movies['year'] = pd.to_datetime(movies['release_date'], errors='coerce').dt.year
    movies['year'] = movies['year'].fillna(0).astype(int)

    return movies[['id', 'title', 'genres_display', 'year', 'vote_average',
                    'overview_text', 'text_overview', 'text_genres',
                    'text_keywords', 'text_cast', 'text_crew']].copy()


if __name__ == '__main__':
    raw = load_raw_data()
    df = preprocess(raw)
    output_path = os.path.join(DATA_DIR, 'preprocessed_data')
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved {len(df)} movies to {output_path}")
