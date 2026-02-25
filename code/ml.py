import os
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), 'embeddings.pkl')
MODEL_NAME = 'all-MiniLM-L6-v2'
TEXT_COLUMNS = {
    'overview': 'text_overview',
    'genres': 'text_genres',
    'keywords': 'text_keywords',
    'cast': 'text_cast',
    'crew': 'text_crew',
}


def build_embeddings(df):
    model = SentenceTransformer(MODEL_NAME)
    embeddings = {}
    for key, col in TEXT_COLUMNS.items():
        print(f"Encoding {key}...")
        embeddings[key] = model.encode(
            df[col].fillna('').tolist(), show_progress_bar=True
        )
    return embeddings


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(DATA_DIR, 'preprocessed_data'))
    embeddings = build_embeddings(df)
    joblib.dump(embeddings, EMBEDDINGS_PATH)
    print(f"Saved embeddings to {EMBEDDINGS_PATH}")
