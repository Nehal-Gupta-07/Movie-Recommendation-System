import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

new_df = pd.read_csv(r"../data/preprocessed_data")

embeddings = joblib.load("embeddings.pkl")

def recommend(movie):
    movie_index = new_df[new_df['title']==movie].index[0]
    movie_embedding = embeddings[movie_index].reshape(1, -1)
    scores = cosine_similarity(movie_embedding, embeddings)[0]
    movies_list = sorted(list(enumerate(scores)),reverse=True,key = lambda x:x[1])[1:6]

    recommended_movies = []
    
    for i in movies_list:
        recommended_movies.append(new_df.iloc[i[0]].title)
    
    return recommended_movies

st.title("Movie Recommender System")

selected_movie_name = st.selectbox(
'Movies List',
new_df['title'].values
)

if st.button('Recommend'):
    recommendations = recommend(selected_movie_name)
    for i in recommendations:
        st.write(i)
