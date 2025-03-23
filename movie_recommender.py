#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import gradio as gr
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import save_npz
import faiss
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import load_npz




# In[2]:


# Load Movies & Ratings datasets
movies = pd.read_csv('data/movie.csv')
ratings = pd.read_csv('data/rating.csv')

# Display the first few rows
print(movies.head())
print(ratings.head())


# In[3]:


print("Movies Columns:\n", movies.columns)
print("\nRatings Columns:\n", ratings.columns)


# In[4]:


print(movies.isnull().sum())
print(ratings.isnull().sum())


# In[5]:


movies.duplicated().sum()
ratings.duplicated().sum()


# In[6]:


ratings = ratings.drop(columns=['timestamp'])
print(ratings.head())


# In[7]:


def convert_genres(genre_str):
    return genre_str.split('|') if isinstance(genre_str, str) else []

movies['genres'] = movies['genres'].apply(convert_genres)
print(movies[['title', 'genres']].head())


# In[8]:


print("Unique Users:", ratings['userId'].nunique())
print("Unique Movies:", movies['title'].nunique())


# In[143]:




# Load ratings from dataset
ratings_df = movies[['userId', 'movieId', 'rating']]

# Split into 80% train, 20% test
train_data, test_data = train_test_split(ratings_df, test_size=0.2, random_state=42)

print(f" Train Set: {train_data.shape}, Test Set: {test_data.shape}")


# In[9]:


# Count how many times each user has rated a movie
user_rating_counts = ratings['userId'].value_counts()

# Keep only users who have rated 50+ movies
active_users = user_rating_counts[user_rating_counts >= 100].index

# Filter ratings for these active users
filtered_ratings = ratings[ratings['userId'].isin(active_users)]

print("Filtered Users:", filtered_ratings['userId'].nunique())


# In[10]:


# Count how many ratings each movie has
movie_rating_counts = ratings['movieId'].value_counts()

# Keep only movies that have 50+ ratings
popular_movies = movie_rating_counts[movie_rating_counts >= 100].index

# Filter ratings for these popular movies
filtered_ratings = filtered_ratings[filtered_ratings['movieId'].isin(popular_movies)]

print("Filtered Movies:", filtered_ratings['movieId'].nunique())


# In[11]:


# Create a mapping of movieId to a sequential index
movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(filtered_ratings['movieId'].unique())}
user_id_to_index = {user_id: idx for idx, user_id in enumerate(filtered_ratings['userId'].unique())}

# Convert userId and movieId to sequential indices
filtered_ratings['user_idx'] = filtered_ratings['userId'].map(user_id_to_index)
filtered_ratings['movie_idx'] = filtered_ratings['movieId'].map(movie_id_to_index)

# Create sparse matrix
sparse_matrix = csr_matrix(
    (filtered_ratings['rating'], (filtered_ratings['user_idx'], filtered_ratings['movie_idx'])),
    shape=(len(user_id_to_index), len(movie_id_to_index))
)

print("Sparse Matrix Shape:", sparse_matrix.shape)




# In[12]:


# Extract ratings as a NumPy array
ratings_array = filtered_ratings['rating'].values.reshape(-1, 1)

# Initialize the MinMaxScaler (scales values between 0 and 1)
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit and transform ratings
normalized_ratings = scaler.fit_transform(ratings_array)

# Replace original ratings with normalized values
filtered_ratings['normalized_rating'] = normalized_ratings.flatten()

# Print updated dataset
print(filtered_ratings[['userId', 'movieId', 'rating', 'normalized_rating']].head())


# In[13]:


# Merge movies.csv and filtered_ratings.csv on 'movieId'
movies_with_ratings = filtered_ratings.merge(movies[['movieId', 'title', 'genres']], on='movieId', how='left')

# Display the merged dataset
print(movies_with_ratings.head())


# In[14]:


movies_with_ratings.to_csv('data/movies_merged.csv', index=False)
print("Merged dataset saved successfully!")


# In[15]:



# Save sparse matrix in a compressed format
save_npz('data/user_movie_sparse_matrix.npz', sparse_matrix)

print("Sparse matrix saved successfully!")


# In[16]:


# Load merged movies dataset
movies_df = pd.read_csv("data/movies_merged.csv")

# Show first few rows
print(movies_df.head())


# In[ ]:





# In[17]:




# Load sparse user-movie ratings matrix
user_movie_sparse = load_npz("data/user_movie_sparse_matrix.npz")


# Convert to compressed sparse row format (CSR) for efficient computation
user_movie_csr = csr_matrix(user_movie_sparse)

print("User-Movie Sparse Matrix Shape:", user_movie_csr.shape)


# In[18]:


# Reduce dimensions using SVD for better efficiency
svd = TruncatedSVD(n_components=100)  # Reduce to 100 latent features
user_reduced = svd.fit_transform(user_movie_csr)

# Create FAISS index
d = user_reduced.shape[1]  # Feature dimensions (100 after SVD)
index = faiss.IndexFlatIP(d)  # Inner Product similarity (cosine equivalent)
index.add(user_reduced)  # Add user embeddings to FAISS index

print("FAISS index created with", index.ntotal, "users.")


# In[19]:


def recommend_movies_user_based(user_id, num_recommendations=5):
    """
    Recommend movies based on similar users' preferences.
    """
    query_vector = user_reduced[user_id].reshape(1, -1)  # Get user vector
    distances, indices = index.search(query_vector, 6)  # Find 5 similar users

    top_similar_users = indices.flatten()[1:]  # Exclude the user itself

    # Get ratings from similar users
    similar_users_ratings = user_movie_csr[top_similar_users].mean(axis=0)

    # Convert to Pandas Series (FIX: Ensure it's 1D)
    movie_recommendations = pd.Series(similar_users_ratings.A1)  

    # Get top movie recommendations
    return movie_recommendations.nlargest(num_recommendations).index.tolist()

# Example: Get recommendations for user 1
print(recommend_movies_user_based(1))


# In[20]:


# Reduce dimensions for items (movies)
item_reduced = svd.fit_transform(user_movie_csr.T)  # Transpose for movie-based filtering

# Create FAISS index for item-item similarity
index_item = faiss.IndexFlatIP(item_reduced.shape[1])
index_item.add(item_reduced)

print("FAISS index created for movies.")


# In[21]:


def recommend_movies_item_based(movie_id, num_recommendations=5):
    """
    Recommend similar movies based on ratings.
    """
    query_vector = item_reduced[movie_id].reshape(1, -1)
    distances, indices = index_item.search(query_vector, num_recommendations + 1)

    return indices.flatten()[1:].tolist()  # Exclude the input movie

# Example: Get similar movies for movie ID 1
print(recommend_movies_item_based(1))



# In[22]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Fill NaN genres with an empty string
movies_df['genres'] = movies_df['genres'].fillna('')

# Convert genres into a TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['genres'])

print("TF-IDF Matrix Shape (Genres):", tfidf_matrix.shape)


# In[23]:


# Convert sparse TF-IDF matrix to dense format (required for FAISS)
tfidf_dense = tfidf_matrix.toarray().astype('float32')  # Convert to float32 for FAISS

print("TF-IDF Matrix Shape (Dense):", tfidf_dense.shape)



# In[24]:


# Define FAISS index for similarity search
d = tfidf_dense.shape[1]  # Number of features
index = faiss.IndexFlatIP(d)  # Inner product similarity (equivalent to cosine)
index.add(tfidf_dense)  # Add movie vectors to the index

print("FAISS index created with", index.ntotal, "movies.")


# In[88]:


# Create a mapping from movie titles to indices
# Strip spaces and remove special characters from titles
movies_df['title'] = movies_df['title'].astype(str).str.strip()

# Update the title_to_index mapping
title_to_index = pd.Series(movies_df.index, index=movies_df['title']).to_dict()


# In[92]:


def recommend_movies_content_faiss(title, num_recommendations=5):
    """
    Recommend movies based on FAISS genre similarity.
    Removes duplicate recommendations.
    """
    # Step 3.1: Check if title exists
    if title not in title_to_index:
        # Try finding a close match
        matching_titles = [t for t in title_to_index.keys() if title.lower() in t.lower()]
        if not matching_titles:
            return f"Movie '{title}' not found in dataset."
        title = matching_titles[0]  # Use first close match
    
    # Step 3.2: Retrieve Movie Index
    movie_idx = title_to_index[title]

    # Step 3.3: Ensure Query Vector Shape Matches FAISS Index
    query_vector = tfidf_dense[movie_idx].reshape(1, -1).astype('float32')  

    # Step 3.4: Search for Top Similar Movies
    distances, indices = index.search(query_vector, num_recommendations * 2)  # Fetch extra results to filter out duplicates

    # Step 3.5: Convert indices to movie titles & Remove Duplicates
    unique_titles = []
    seen_titles = set()
    
    for idx in indices.flatten():
        movie_title = movies_df['title'].iloc[idx]
        if movie_title not in seen_titles and movie_title != title:  # Exclude original movie
            unique_titles.append(movie_title)
            seen_titles.add(movie_title)
        
        if len(unique_titles) == num_recommendations:  # Stop once we have enough unique movies
            break

    return unique_titles

# Example: Get similar movies for "Up (2009)"
print(recommend_movies_content_faiss("Up (2009)"))


# In[157]:


# Define weights for the hybrid model
ALPHA = 0.6  # Weight for Collaborative Filtering
BETA = 1 - ALPHA  # Weight for Content-Based Filtering

def recommend_movies_hybrid(user_id, title="", num_recommendations=5):
    """
    Hybrid Recommendation System:
    Combines User-Based Collaborative Filtering & Content-Based Filtering.
    
    Parameters:
    - user_id: ID of the user for collaborative filtering
    - title: Movie title for content-based filtering
    - num_recommendations: Number of recommendations to return
    
    Returns:
    - List of recommended movie titles
    """

    # Step 1: Get User-Based Collaborative Filtering Recommendations
    try:
        similar_users = user_movie_csr[user_id].sort_values(ascending=False)[1:6]  # Top 5 similar users
        similar_users_ratings = user_movie_sparse[similar_users.index].mean(axis=0)  # Get average rating
        
        # Convert to Pandas Series & normalize scores
        collaborative_scores = pd.Series(similar_users_ratings.A1, index=movies_df.index).fillna(0)
        collaborative_scores = (collaborative_scores - collaborative_scores.min()) / (collaborative_scores.max() - collaborative_scores.min())  # Normalize
    except:
        collaborative_scores = pd.Series(0, index=movies_df.index)  # If no collaborative data, set to 0

    # Step 2: Get Content-Based Filtering Recommendations
    
    if title=="" :
        content_scores = pd.Series(0, index=movies_df.index)  # If no content data, set to 0
    else :   
        content_similar_movies = recommend_movies_content_faiss(title, num_recommendations * 2)  # Get more to avoid duplicates
        content_scores = pd.Series(1, index=[title_to_index[movie] for movie in content_similar_movies if movie in title_to_index])
    

    # Step 3: Merge Scores Using Weighted Hybrid Formula
    hybrid_scores = ALPHA * collaborative_scores + BETA * content_scores

    # Step 4: Get Top Movie Recommendations
    top_movies = hybrid_scores.nlargest(num_recommendations).index
    return movies_df['title'].iloc[top_movies].tolist()

# Example: Get hybrid recommendations for user 1 & movie "Up (2009)"
print(recommend_movies_hybrid(user_id=1))


# In[97]:


# Get unique user IDs from ratings dataset
user_ids = sorted(movies_df['userId'].unique().tolist())  # Ensure sorted order


# In[123]:


import ast  # For converting string lists into actual lists

# Initialize an empty set to store unique genres
all_genres = set()

# Iterate over each row in the 'genres' column
for genre_str in movies_df['genres'].dropna():
    try:
        genres = ast.literal_eval(genre_str)  # Convert string to list if needed
    except (ValueError, SyntaxError):
        genres = genre_str.split('|')  # If already formatted, split by "|"

    all_genres.update(genres)  # Add individual genres to the set

# Convert to sorted list & add "All" option
unique_genre = ["All"] + sorted(all_genres)

print(unique_genre)  # Debugging: Check if genres are correct



# In[127]:


# Updated Mood-Genre Mapping for Sentiment Analysis
mood_genre_mapping = {
    "Happy": ["Comedy", "Animation", "Musical", "Fantasy", "Children"],
    "Sad": ["Drama", "War", "Film-Noir"],
    "Excited": ["Action", "Adventure", "Thriller", "Sci-Fi", "Crime"],
    "Neutral": ["Documentary", "Western", "IMAX"],
    "Dark": ["Horror", "Mystery"]
}


# In[135]:


import ast

def gradio_recommend(user_id, movie_title, genre, user_mood, num_recommendations=5):
    """
    Flexible Recommendation System supporting Mood, Genre, and Title-based filtering.
    """
    # Get recommendations (Default: Hybrid Model)
    if movie_title:
        recommendations = recommend_movies_hybrid(user_id=int(user_id), title=movie_title, num_recommendations=num_recommendations)
    else:
        recommendations = recommend_movies_hybrid(user_id=int(user_id), num_recommendations=num_recommendations)  # No title provided

    # Apply Genre Filtering (if selected)
    if genre and genre != "All":
        recommendations = [
            movie for movie in recommendations 
            if genre in ast.literal_eval(movies_df[movies_df['title'] == movie]['genres'].values[0])
        ]

    # Apply Mood Filtering (if selected)
    if user_mood in mood_genre_mapping:
        mood_genres = set(mood_genre_mapping[user_mood])
        recommendations = [
            movie for movie in recommendations
            if any(g in mood_genres for g in ast.literal_eval(movies_df[movies_df['title'] == movie]['genres'].values[0]))
        ]
    
    return recommendations[:num_recommendations] if recommendations else ["No recommendations found. Try different inputs."]


# In[145]:


# Extract test data from your dataset (actual user-movie interactions)
test_users = test_data['userId'].unique()  # Users in the test set

# Get actual movie preferences per user
actual_movies_per_user = {
    user: test_data[test_data['userId'] == user]['movieId'].tolist()
    for user in test_users
}

print(f"✅ Test Users: {len(test_users)}")


# In[159]:


# Function to generate recommendations for each user
def evaluate_model(test_users, actual_movies_per_user, num_recommendations=5):
    results = []

    for user_id in test_users:
        # Generate recommendations using YOUR custom model
        recommended_movies = recommend_movies_hybrid(user_id=user_id, num_recommendations=num_recommendations)

        # Store actual and predicted values
        actual_movies = actual_movies_per_user[user_id]

        results.append((user_id, actual_movies, recommended_movies))

    return results

# Run evaluation
evaluation_results = evaluate_model(test_users, actual_movies_per_user)
print(f"✅ Evaluated {len(evaluation_results)} users!")



