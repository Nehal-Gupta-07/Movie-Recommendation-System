import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer

new_df = pd.read_csv(r"../data/preprocessed_data")

model = SentenceTransformer('all-MiniLM-L6-v2')

columns = ['text_overview', 'text_genres', 'text_keywords', 'text_cast', 'text_crew']
embeddings = {}
for col in columns:
    key = col.replace('text_', '')
    print(f"Encoding {key}...")
    embeddings[key] = model.encode(new_df[col].fillna('').tolist(), show_progress_bar=True)

joblib.dump(embeddings, 'embeddings.pkl')
