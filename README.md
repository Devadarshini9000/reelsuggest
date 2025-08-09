## 🎥 reelsuggest - Movie Recommendation System

The **reelsuggest - Movie Recommendation System** is a **content-based recommendation engine** that suggests movies similar to a given title using **TF-IDF vectorization** and cosine similarity.
It processes movie metadata, performs exploratory data analysis, and serves recommendations via a **Flask web application** and a standalone Python prediction script.

## 🚀 Features

- Data Preprocessing – Cleans and merges movie metadata, credits, keywords, ratings, and links
- Genre & Keyword Extraction – Parses JSON-like fields into structured lists for analysis
- Actor & Director Information – Extracts top cast and director names from credits
- Exploratory Data Analysis (EDA) – Visualizes ratings distribution, genre frequency, top-rated movies, popularity trends, and keyword clouds
- Content-Based Recommendation Model – Uses TF-IDF on genres and keywords to find similar movies
- Cosine Similarity Search – Quickly retrieves top N similar movies to a given title
- Flask Web Interface – Search a movie and get 10 similar recommendations instantly
- Search Suggestions – Offers similar title suggestions if exact match is not found
- Pickle Model Storage – Saves preprocessed movie data and TF-IDF matrix for quick loading

## 📂 Project Structure

- ├── app.py                        # Flask web app for movie recommendations
- ├── data_prep.py                  # Preprocessing script for movie datasets
- ├── EDA.py                        # Exploratory Data Analysis with visualizations
- ├── predict.py                    # Standalone script for generating recommendations
- ├── movie_recommendation_model.pkl# Pickled model data with TF-IDF matrix
- ├── processed_movies.csv          # Cleaned and merged movie dataset
- ├── README.md                     # Project documentation

## 🛠 Tech Stack

- Python
- Pandas – Data cleaning and merging
- Scikit-learn – TF-IDF vectorization & cosine similarity
- Matplotlib & Seaborn – Data visualization
- WordCloud – Keyword cloud generation
- Flask – Web application framework
- Pickle – Model persistence

## 🔍 How It Works

1. Data Preprocessing (data_prep.py)
Loads multiple datasets (movies_metadata.csv, credits.csv, keywords.csv, links.csv, ratings.csv)
Extracts genres, keywords, cast, and crew information
Aggregates ratings to compute average scores and number of ratings
Saves the merged and cleaned dataset as processed_movies.csv

2. Exploratory Data Analysis (EDA.py)
Visualizes rating distribution, top genres, highest-rated movies, most popular movies, keyword clouds, and correlation heatmaps

3. Recommendation Model (predict.py & movie_recommendation_model.pkl)
Uses TF-IDF on genres and keywords
Computes cosine similarity between all movie vectors
Retrieves top N most similar movies to a given title

4. Web Application (app.py)
User inputs a movie title through a web form
Model returns 10 similar movies with similarity score, genres, and average rating
If movie is not found, suggests closest matching titles

## 🖥 How to Run

### 1️⃣ Clone the Repo
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system

### 2️⃣ Install Requirements
pip install pandas scikit-learn flask matplotlib seaborn wordcloud numpy

### 3️⃣ Preprocess Data
python data_prep.py

### 4️⃣ Run EDA (Optional)
python EDA.py

### 5️⃣ Run Flask App
python app.py

### 6️⃣ Use Standalone Prediction Script
python predict.py

## 💡 Use Cases
- Personalized Movie Discovery – Suggests movies tailored to user’s preferences
- Streaming Platforms – Integrate recommendations into OTT platforms for better engagement
- Film Enthusiast Tools – Explore movies similar to favorites by genre and style
- Content Curation – Helps curators and critics discover lesser-known but similar titles
- Title Search Assistance – Suggests alternatives if exact title match fails

## 👩‍💻 Author
**Devadarshini P**  
[🔗 LinkedIn](https://www.linkedin.com/in/devadarshini-p-707b15202/)  
[💻 GitHub](https://github.com/Devadarshini9000)

"Discover the next movie you’ll love – let similarity be your guide." 🍿

<img width="813" height="446" alt="image" src="https://github.com/user-attachments/assets/90147c35-7e35-47a7-8992-89ae74e0d52f" />
<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/db731681-329d-4b80-8697-b0b846ae189a" />
<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/919bcb35-56f9-4183-b705-4ed6fdf90ba4" />
<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/5c6564d3-e16f-477e-8f4b-11104cbcffe1" />





