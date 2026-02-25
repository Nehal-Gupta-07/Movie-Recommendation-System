import streamlit as st
import pandas as pd
import numpy as np
import joblib
from urllib.parse import quote
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

EMBEDDING_KEYS = ['overview', 'genres', 'keywords', 'cast', 'crew']


@st.cache_data
def load_data():
    df = pd.read_csv(r"../data/preprocessed_data")
    df['genres_display'] = df['genres_display'].fillna('')
    df['overview_text'] = df['overview_text'].fillna('')
    df['vote_average'] = df['vote_average'].fillna(0.0)
    df['year'] = df['year'].fillna(0).astype(int)
    return df


@st.cache_data
def load_embeddings():
    return joblib.load("embeddings.pkl")


new_df = load_data()
embeddings = load_embeddings()


def recommend(movie, weights, count=5, genre_filter=None, year_range=None):
    movie_index = new_df[new_df['title'] == movie].index[0]

    weight_sum = sum(weights.values())
    if weight_sum == 0:
        norm_weights = {k: 1.0 / len(weights) for k in weights}
    else:
        norm_weights = {k: v / weight_sum for k, v in weights.items()}

    combined_scores = np.zeros(len(new_df))
    for key in EMBEDDING_KEYS:
        movie_emb = embeddings[key][movie_index].reshape(1, -1)
        sim = cosine_similarity(movie_emb, embeddings[key])[0]
        combined_scores += norm_weights[key] * sim

    scored_indices = sorted(
        list(enumerate(combined_scores)), reverse=True, key=lambda x: x[1]
    )

    results = []
    for idx, score in scored_indices:
        if idx == movie_index:
            continue

        row = new_df.iloc[idx]

        if genre_filter:
            movie_genres = [g.strip() for g in row['genres_display'].split(',') if g.strip()]
            if not any(g in genre_filter for g in movie_genres):
                continue

        if year_range:
            if row['year'] != 0 and not (year_range[0] <= row['year'] <= year_range[1]):
                continue

        results.append({
            'title': row['title'],
            'score': score,
            'genres': row['genres_display'],
            'year': row['year'],
            'rating': row['vote_average'],
            'overview': row['overview_text'],
        })

        if len(results) >= count:
            break

    return results


st.markdown("""
<style>
    .movie-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1rem;
        height: 100%;
        border: 1px solid #30475e;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .movie-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    a.movie-title {
        display: block;
        font-size: 1.05rem;
        font-weight: 700;
        color: #e0e0e0;
        margin: 0.5rem 0 0.3rem 0;
        line-height: 1.3;
        text-decoration: none;
    }
    a.movie-title:hover {
        color: #e94560;
        text-decoration: underline;
    }
    .movie-meta {
        font-size: 0.85rem;
        color: #a0a0a0;
        margin: 0.15rem 0;
    }
    .similarity-badge {
        display: inline-block;
        background: #e94560;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.4rem;
    }
    .rating-badge {
        display: inline-block;
        background: #f5c518;
        color: #1a1a2e;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎬 Movie Recommender System")
st.caption("Select a movie you like, tweak the filters, and discover similar films.")

# --- Sidebar filters ---
all_genres = sorted(set(
    g.strip()
    for genres_str in new_df['genres_display'].dropna()
    for g in genres_str.split(',')
    if g.strip()
))

valid_years = new_df[new_df['year'] > 0]['year']
min_year = int(valid_years.min()) if len(valid_years) > 0 else 1900
max_year = int(valid_years.max()) if len(valid_years) > 0 else 2025

with st.sidebar:
    st.header("Filters")

    num_recommendations = st.slider("Number of recommendations", 1, 20, 5)

    selected_genres = st.multiselect("Filter by genre", all_genres)

    year_range = st.slider("Release year range", min_year, max_year, (min_year, max_year))

    st.markdown("---")
    st.header("Similarity Weights")
    st.caption("Adjust how much each feature influences recommendations.")

    w_overview = st.slider("Plot overview", 0.0, 1.0, 0.4, 0.05)
    w_genres = st.slider("Genres", 0.0, 1.0, 0.2, 0.05)
    w_keywords = st.slider("Keywords", 0.0, 1.0, 0.2, 0.05)
    w_cast = st.slider("Cast", 0.0, 1.0, 0.1, 0.05)
    w_crew = st.slider("Director", 0.0, 1.0, 0.1, 0.05)

    feature_weights = {
        'overview': w_overview,
        'genres': w_genres,
        'keywords': w_keywords,
        'cast': w_cast,
        'crew': w_crew,
    }

# --- Movie selection ---
selected_movie_name = st.selectbox("Choose a movie you like", new_df['title'].values)

if st.button("🔍 Get Recommendations", type="primary", use_container_width=True):
    genre_filter = selected_genres if selected_genres else None
    year_filter = year_range if year_range != (min_year, max_year) else None

    recommendations = recommend(
        selected_movie_name,
        weights=feature_weights,
        count=num_recommendations,
        genre_filter=genre_filter,
        year_range=year_filter,
    )

    if not recommendations:
        st.warning("No movies found matching your filters. Try broadening the genre or year range.")
    else:
        st.markdown("---")
        st.subheader(f"Movies similar to *{selected_movie_name}*")

        cols_per_row = min(5, len(recommendations))

        for row_start in range(0, len(recommendations), cols_per_row):
            row_items = recommendations[row_start:row_start + cols_per_row]
            cols = st.columns(cols_per_row)

            for col, movie in zip(cols, row_items):
                similarity_pct = f"{movie['score'] * 100:.0f}%"
                year_str = str(movie['year']) if movie['year'] > 0 else "N/A"
                wiki_url = f"https://en.wikipedia.org/wiki/{quote(movie['title'].replace(' ', '_'))}"

                with col:
                    st.markdown(
                        f"""
                        <div class="movie-card">
                            <a href="{wiki_url}" target="_blank" class="movie-title">{movie['title']}</a>
                            <p class="movie-meta">
                                <span class="rating-badge">⭐ {movie['rating']:.1f}</span>
                                &nbsp; {year_str}
                            </p>
                            <p class="movie-meta">{movie['genres']}</p>
                            <span class="similarity-badge">{similarity_pct} match</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
