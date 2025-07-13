import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load processed data
movies = pd.read_csv("processed_movies.csv")

# Display basic info
print("🔹 Dataset Overview:")
print(movies.info())

# Check for missing values
print("\n🔹 Missing Values:")
print(movies.isnull().sum())

# Display first few rows
print("\n🔹 Sample Data:")
print(movies.head())

# Convert genres list to string (for visualization)
movies['genres'] = movies['genres'].fillna('[]').apply(lambda x: ', '.join(eval(x)) if isinstance(x, str) else '')

# -------------------------------
# 📊 1️⃣ Distribution of Ratings
# -------------------------------
plt.figure(figsize=(8,5))
sns.histplot(movies['avg_rating'].dropna(), bins=20, kde=True, color='blue')
plt.xlabel("Average Rating")
plt.ylabel("Frequency")
plt.title("Distribution of Movie Ratings")
plt.show()

# -------------------------------
# 🎭 2️⃣ Top 10 Most Common Genres
# -------------------------------
all_genres = ', '.join(movies['genres'].dropna()).split(', ')
genre_counts = pd.Series(all_genres).value_counts()

plt.figure(figsize=(10,5))
sns.barplot(x=genre_counts.index[:10], y=genre_counts.values[:10], palette="viridis")
plt.xticks(rotation=45)
plt.xlabel("Genre")
plt.ylabel("Count")
plt.title("Top 10 Most Common Genres")
plt.show()

# -------------------------------
# 🎬 3️⃣ Top 10 Highest Rated Movies
# -------------------------------
top_rated = movies.sort_values(by='avg_rating', ascending=False).head(10)
plt.figure(figsize=(10,5))
sns.barplot(y=top_rated['title'], x=top_rated['avg_rating'], palette="magma")
plt.xlabel("Average Rating")
plt.ylabel("Movie Title")
plt.title("Top 10 Highest Rated Movies")
plt.show()

# -------------------------------
# 🌟 4️⃣ Top 10 Most Popular Movies (by vote count)
# -------------------------------
top_popular = movies.sort_values(by='num_ratings', ascending=False).head(10)
plt.figure(figsize=(10,5))
sns.barplot(y=top_popular['title'], x=top_popular['num_ratings'], palette="coolwarm")
plt.xlabel("Number of Ratings")
plt.ylabel("Movie Title")
plt.title("Top 10 Most Popular Movies")
plt.show()

# -------------------------------
# 🔥 5️⃣ WordCloud of Most Common Keywords
# -------------------------------
movies['keywords'] = movies['keywords'].fillna('[]').apply(lambda x: ', '.join(eval(x)) if isinstance(x, str) else '')
keyword_text = ' '.join(movies['keywords'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Set2').generate(keyword_text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud of Movie Keywords")
plt.show()

# -------------------------------
# 🔄 6️⃣ Correlation Heatmap
# -------------------------------
plt.figure(figsize=(8,6))
sns.heatmap(movies[['budget', 'revenue', 'avg_rating', 'num_ratings']].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
