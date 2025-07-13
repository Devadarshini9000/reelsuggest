import pandas as pd
import numpy as np
import nltk
import pickle
import ast
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK stopwords
nltk.download("stopwords")
from nltk.corpus import stopwords

# Load movie dataset
movies = pd.read_csv(r"D:\UNM\movie\processed_movies.csv")

print("Dataset columns:", movies.columns.tolist())
print(f"Total movies: {len(movies)}")

# Function to safely parse string lists
def parse_list_string(list_str):
    if isinstance(list_str, str):
        try:
            parsed = ast.literal_eval(list_str)
            if isinstance(parsed, list):
                return ' '.join(parsed)
            return str(parsed)
        except (SyntaxError, ValueError):
            return list_str
    return ''

# Create a combined features column for content-based filtering
movies['genres_str'] = movies['genres'].apply(parse_list_string)

# Process keywords
if 'keywords' in movies.columns:
    movies['keywords_str'] = movies['keywords'].apply(parse_list_string)
else:
    movies['keywords_str'] = ''

# Process cast - take top 3 cast members if possible
if 'cast' in movies.columns:
    def parse_cast(cast_str):
        if isinstance(cast_str, str):
            try:
                cast_list = ast.literal_eval(cast_str)
                if isinstance(cast_list, list):
                    return ' '.join(cast_list[:3])
                return str(cast_list)
            except (SyntaxError, ValueError):
                return cast_str
        return ''
    
    movies['cast_str'] = movies['cast'].apply(parse_cast)
else:
    movies['cast_str'] = ''

# Create combined features
movies['combined_features'] = movies['title'] + ' ' + movies['genres_str'] + ' ' + movies['keywords_str'] + ' ' + movies['cast_str']
movies['combined_features'] = movies['combined_features'].fillna('').str.lower()

print("\nSample combined features:")
print(movies['combined_features'].head())

# TF-IDF Vectorization on combined features
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["combined_features"])

print(f"\nTF-IDF matrix shape: {tfidf_matrix.shape}")

# Instead of computing the full similarity matrix, save just the TF-IDF matrix
# We'll compute similarities on-demand in the prediction script
model_data = {
    "movies": movies[['id', 'title', 'genres', 'release_date', 'movieId', 'avg_rating']],
    "tfidf_vectorizer": tfidf,
    "tfidf_matrix": tfidf_matrix,
}

with open("movie_recommendation_model.pkl", "wb") as file:
    pickle.dump(model_data, file)

print("\n✅ Training Completed & Model Saved as 'movie_recommendation_model.pkl' 🎬")
print("Note: Similarities will be computed on-demand during prediction to save memory")