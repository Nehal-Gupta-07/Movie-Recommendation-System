import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer

new_df = pd.read_csv(r"../data/preprocessed_data")

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(new_df['tags'].tolist(), show_progress_bar=True)

joblib.dump(embeddings, 'embeddings.pkl')
