# Movie Recommender System

A content-based movie recommendation system built with Python and Streamlit. It uses separate sentence-transformer embeddings for plot, genres, keywords, cast, and director, then computes a weighted cosine similarity to suggest similar movies.

---

## Project Overview

The system processes the TMDB 5000 Movies dataset and builds per-feature semantic embeddings using the `all-MiniLM-L6-v2` sentence transformer model. At recommendation time, cosine similarity is computed independently for each feature (overview, genres, keywords, cast, director) and combined with user-adjustable weights, giving fine-grained control over what drives the recommendations.

Movie titles in the results link directly to their Wikipedia pages.

---

## Tech Stack

| Layer       | Technology                                          |
|-------------|-----------------------------------------------------|
| Frontend    | Streamlit                                           |
| Backend     | Python                                              |
| ML Model    | Sentence Transformers (`all-MiniLM-L6-v2`)          |
| Libraries   | pandas, scikit-learn, NumPy, joblib                 |
| Algorithm   | Content-based filtering with weighted cosine similarity |

---

## Project Structure

```
Movie-Recommender/
├── code/
│   ├── helper_functions.py   # Parsing utilities for JSON columns
│   ├── preprocessing.py      # Cleans raw data, builds per-feature text columns
│   ├── ml.py                 # Encodes 5 text columns into separate embeddings
│   └── app.py                # Streamlit UI with filters and weight sliders
├── data/
│   ├── tmdb_5000_movies.csv  # Raw movies dataset
│   ├── tmdb_5000_credits.csv # Raw credits dataset
│   └── preprocessed_data     # Generated CSV (created by preprocessing.py)
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Nehal-Gupta-07/Movie-Recommendation-System.git
cd Movie-Recommendation-System
```

### 2. (Optional) Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate    # macOS / Linux
venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the data

Place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` inside the `data/` folder, then run from the `code/` directory:

```bash
cd code
python preprocessing.py
python ml.py
```

This generates the preprocessed CSV and the `embeddings.pkl` file.

### 5. Run the app

```bash
streamlit run app.py
```

---

## How It Works

1. **Preprocessing** (`preprocessing.py`) -- Merges movies and credits data, extracts genres, keywords, top 5 cast, and director into separate text columns. Also retains metadata like rating, year, and overview for display.

2. **Embedding** (`ml.py`) -- Encodes each of the 5 text columns independently using the `all-MiniLM-L6-v2` sentence transformer and saves them as a dictionary in `embeddings.pkl`.

3. **Recommendation** (`app.py`) -- For a selected movie, computes cosine similarity across all 5 embedding spaces and combines them with a weighted sum:

   ```
   score = w1 * sim(overview) + w2 * sim(genres) + w3 * sim(keywords) + w4 * sim(cast) + w5 * sim(director)
   ```

   Weights are adjustable via sidebar sliders. Results can also be filtered by genre and release year.

---

## Features

- **Weighted similarity** -- Five sidebar sliders let you control how much plot, genres, keywords, cast, and director each influence the results.
- **Genre filter** -- Multi-select to narrow recommendations to specific genres.
- **Year range filter** -- Slider to limit recommendations to a release year window.
- **Adjustable count** -- Choose between 1 and 20 recommendations.
- **Wikipedia links** -- Click any recommended movie title to open its Wikipedia page.

---

## Contributing

1. Fork this repository
2. Create a new branch (`git checkout -b feature/awesome-feature`)
3. Make your changes
4. Commit (`git commit -m "Add feature"`)
5. Push (`git push origin feature/awesome-feature`)
6. Open a Pull Request
