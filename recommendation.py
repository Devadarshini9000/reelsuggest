import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD
from surprise import Dataset, Reader

# Load preprocessed movies dataset
movies = pd.read_csv("processed_movies.csv")

# Load trained SVD model
with open("svd_model.pkl", "rb") as f:
    svd = pickle.load(f)

# Load ratings dataset for user-based predictions
ratings = pd.read_csv("ratings.csv")
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# -----------------------------------------
# 🎭 1️⃣ Predict Movie Rating for a User
# -----------------------------------------
def predict_rating(user_id, movie_title):
    movie_id = movies[movies['title'] == movie_title]['movieId']
    
    if movie_id.empty:
        return "Movie not found!"
    
    movie_id = movie_id.values[0]
    predicted_rating = svd.predict(user_id, movie_id).est
    return f"Predicted Rating for '{movie_title}' by User {user_id}: {predicted_rating:.2f}"

# -----------------------------------------
# 🔢 2️⃣ Recommend Top Movies for a User
# -----------------------------------------
def recommend_for_user(user_id, top_n=5):
    user_movies = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    
    predictions = [(movie, svd.predict(user_id, movie).est) for movie in movies['movieId'] if movie not in user_movies]
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
    
    movie_ids = [pred[0] for pred in predictions]
    return movies[movies['movieId'].isin(movie_ids)][['title', 'avg_rating']].to_dict(orient='records')

# -----------------------------------------
# 🎬 3️⃣ Recommend Similar Movies (Content-Based)
# -----------------------------------------
movies['genres'] = movies['genres'].fillna('[]').apply(lambda x: ' '.join(eval(x)) if isinstance(x, str) else '')
movies['keywords'] = movies['keywords'].fillna('[]').apply(lambda x: ' '.join(eval(x)) if isinstance(x, str) else '')

movies['content'] = movies['genres'] + " " + movies['keywords']
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['content'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_similar_movies(movie_title, top_n=5):
    if movie_title not in movies['title'].values:
        return ["Movie not found!"]
    
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]

    return movies['title'].iloc[movie_indices].tolist()

# -----------------------------------------
# 🚀 Example Predictions
# -----------------------------------------
print("\n🎬 Predicting Rating for 'Avatar' by User 1:")
print(predict_rating(1, "Avatar"))

print("\n👥 Top 5 Recommendations for User 1:")
print(recommend_for_user(1))

print("\n🎭 Movies Similar to 'Avatar':")
print(recommend_similar_movies("Avatar"))
