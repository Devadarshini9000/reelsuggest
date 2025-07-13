from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the model when the app starts
try:
    with open("movie_recommendation_model.pkl", "rb") as file:
        model_data = pickle.load(file)
    movies = model_data["movies"]
    tfidf_matrix = model_data["tfidf_matrix"]
    print(f"Model loaded successfully. {len(movies)} movies available.")
except Exception as e:
    print(f"Error loading model: {e}")
    movies = pd.DataFrame()
    tfidf_matrix = None

@app.route('/')
def index():
    # Get a list of random movies for the dropdown
    if not movies.empty:
        sample_movies = movies.sample(min(20, len(movies)))['title'].tolist()
    else:
        sample_movies = []
    return render_template('index.html', sample_movies=sample_movies)

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form.get('movie_title')
    
    if not movie_title:
        return jsonify({"error": "Please enter a movie title"}), 400
    
    if movies.empty or tfidf_matrix is None:
        return jsonify({"error": "Model not loaded properly"}), 500
    
    # Find the movie index
    movie_indices = movies[movies['title'].str.lower() == movie_title.lower()].index
    
    if len(movie_indices) == 0:
        # Find similar titles
        similar_titles = movies[movies['title'].str.lower().str.contains(movie_title.lower())]
        if not similar_titles.empty:
            suggestions = similar_titles['title'].head(5).tolist()
            return jsonify({"error": f"Movie '{movie_title}' not found", 
                           "suggestions": suggestions}), 404
        else:
            return jsonify({"error": f"Movie '{movie_title}' not found. No similar titles found."}), 404
    
    movie_idx = movie_indices[0]
    
    # Get the movie's TF-IDF vector
    movie_vector = tfidf_matrix[movie_idx]
    
    # Compute similarities with all movies for this specific movie
    sim_scores = cosine_similarity(movie_vector, tfidf_matrix).flatten()
    
    # Get the indices of the top 10 similar movies
    # Exclude the movie itself (which would have similarity=1)
    similar_movie_indices = sim_scores.argsort()[::-1][1:11]
    
    # Get the similarity scores
    similar_movie_scores = sim_scores[similar_movie_indices]
    
    # Create a list of recommendations
    recommendations = []
    for i, idx in enumerate(similar_movie_indices):
        recommendations.append({
            'title': movies.iloc[idx]['title'],
            'similarity': f"{similar_movie_scores[i]:.2f}",
            'genres': movies.iloc[idx]['genres'],
            'avg_rating': f"{movies.iloc[idx]['avg_rating']:.2f}" if not pd.isna(movies.iloc[idx]['avg_rating']) else "N/A"
        })
    
    return jsonify({"recommendations": recommendations})

if __name__ == '__main__':
    app.run(debug=True)