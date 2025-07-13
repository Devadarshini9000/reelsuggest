import pandas as pd
import numpy as np
import ast

# Load datasets
movies = pd.read_csv("movies_metadata.csv", low_memory=False)
credits = pd.read_csv("credits.csv")
keywords = pd.read_csv("keywords.csv")
links = pd.read_csv("links.csv")
ratings = pd.read_csv("ratings.csv")

# Convert 'id' to numeric, forcing errors to NaN
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
credits['id'] = pd.to_numeric(credits['id'], errors='coerce')
keywords['id'] = pd.to_numeric(keywords['id'], errors='coerce')

# Preprocessing movies_metadata.csv
movies = movies[['id', 'title', 'genres', 'release_date', 'budget', 'revenue', 'vote_average', 'vote_count']]
movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
movies['budget'] = pd.to_numeric(movies['budget'], errors='coerce')
movies['revenue'] = pd.to_numeric(movies['revenue'], errors='coerce')

# Extract genres from JSON-like structure
def parse_json_column(data):
    try:
        parsed_data = ast.literal_eval(data)
        return [item['name'] for item in parsed_data]
    except (ValueError, SyntaxError):
        return []

movies['genres'] = movies['genres'].fillna('[]').apply(parse_json_column)

# Process 'credits.csv' safely
credits['cast'] = credits['cast'].fillna('[]').apply(lambda x: parse_json_column(x)[:5])  # Top 5 cast members
credits['crew'] = credits['crew'].fillna('[]').apply(lambda x: [d['name'] for d in ast.literal_eval(x) if d['job'] == 'Director'] if isinstance(x, str) else [])

# Process 'keywords.csv'
keywords['keywords'] = keywords['keywords'].fillna('[]').apply(parse_json_column)

# Merge datasets
movies = movies.merge(credits, on='id', how='left')
movies = movies.merge(keywords, on='id', how='left')

# Preprocess ratings.csv
ratings = ratings.drop(columns=['timestamp'])
ratings = ratings.groupby('movieId').agg({'rating': ['mean', 'count']}).reset_index()
ratings.columns = ['movieId', 'avg_rating', 'num_ratings']

# Process 'links.csv'
links['tmdbId'] = pd.to_numeric(links['tmdbId'], errors='coerce')

# Merge movies with links and ratings
movies = movies.merge(links[['tmdbId', 'movieId']], left_on='id', right_on='tmdbId', how='left')
movies = movies.merge(ratings, on='movieId', how='left')

# Handle missing values
movies['avg_rating'] = movies['avg_rating'].fillna(movies['avg_rating'].mean())
movies['num_ratings'] = movies['num_ratings'].fillna(0)

# Save the preprocessed data
movies.to_csv("processed_movies.csv", index=False)

print("✅ Preprocessing completed. Saved as 'processed_movies.csv'.")
