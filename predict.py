import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations(movie_title, model_path="movie_recommendation_model.pkl", top_n=10):
    """
    Get movie recommendations based on similarity to the input movie title
    
    Parameters:
    movie_title (str): Title of the movie to get recommendations for
    model_path (str): Path to the saved model file
    top_n (int): Number of recommendations to return
    
    Returns:
    pd.DataFrame: Top N similar movies with similarity scores
    """
    # Load the saved model
    with open(model_path, "rb") as file:
        model_data = pickle.load(file)
    
    movies = model_data["movies"]
    tfidf_matrix = model_data["tfidf_matrix"]
    
    # Find the movie index
    movie_indices = movies[movies['title'].str.lower() == movie_title.lower()].index
    
    if len(movie_indices) == 0:
        print(f"Movie '{movie_title}' not found. Here are some title suggestions:")
        # Find similar titles
        similar_titles = movies[movies['title'].str.lower().str.contains(movie_title.lower())]
        if not similar_titles.empty:
            print(similar_titles['title'].head(5).tolist())
        else:
            print("No similar titles found.")
        return None
    
    movie_idx = movie_indices[0]
    
    # Get the movie's TF-IDF vector
    movie_vector = tfidf_matrix[movie_idx]
    
    # Compute similarities with all movies for this specific movie
    sim_scores = cosine_similarity(movie_vector, tfidf_matrix).flatten()
    
    # Get the indices of the top N similar movies
    # Exclude the movie itself (which would have similarity=1)
    similar_movie_indices = sim_scores.argsort()[::-1][1:top_n+1]
    
    # Get the similarity scores
    similar_movie_scores = sim_scores[similar_movie_indices]
    
    # Create a DataFrame with recommendations
    recommendations = pd.DataFrame({
        'title': movies.iloc[similar_movie_indices]['title'].values,
        'similarity': similar_movie_scores,
        'genres': movies.iloc[similar_movie_indices]['genres'].values,
        'avg_rating': movies.iloc[similar_movie_indices]['avg_rating'].values,
    })
    
    return recommendations

# Example usage
if __name__ == "__main__":
    movie_title = input("Enter a movie title: ")
    recommendations = get_recommendations(movie_title)
    
    if recommendations is not None:
        print(f"\nTop 10 recommendations for '{movie_title}':")
        pd.set_option('display.max_colwidth', None)
        print(recommendations)
        