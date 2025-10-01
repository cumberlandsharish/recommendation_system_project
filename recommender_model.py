import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np

# --- 1. CONFIGURATION AND DATA LOADING ---
# Path to the MovieLens 100k rating data
DATA_PATH = 'u.data'

# Column names based on MovieLens documentation
COLUMN_NAMES = ['user_id', 'movie_id', 'rating', 'timestamp']

def load_data():
    """Loads the raw data and creates the user-item matrix."""
    try:
        # Load the ratings data (tab-separated)
        df = pd.read_csv(
            DATA_PATH, 
            sep='\t', 
            names=COLUMN_NAMES, 
            encoding='latin-1'
        )
    except FileNotFoundError:
        print(f"Error: Could not find the required data file at {DATA_PATH}.")
        print("Please ensure 'u.data' from the MovieLens 100k dataset is in the project directory.")
        return None, None

    # --- 2. PREPROCESSING: CREATE USER-ITEM MATRIX ---
    # Pivot the table to get the User-Item matrix (users as rows, movies as columns)
    user_movie_matrix = df.pivot_table(
        index='user_id', 
        columns='movie_id', 
        values='rating'
    ).fillna(0) # Fill NaN (unrated movies) with 0

    # --- 3. SPARSITY HANDLING: CONVERT TO CSR MATRIX ---
    # Convert the dense DataFrame to a Compressed Sparse Row (CSR) matrix
    # This is memory-efficient and required for scikit-learn's NearestNeighbors
    sparse_matrix = csr_matrix(user_movie_matrix.values)
    
    # Return the sparse matrix and the DataFrame index/columns for mapping
    return sparse_matrix, user_movie_matrix

# --- 4. MODEL TRAINING ---
def train_model(sparse_matrix):
    """Initializes and trains the k-NN model."""
    # Use 'cosine' similarity metric and 'brute' force search (efficient for sparse data)
    # k=20 is a common value to use for k-NN in collaborative filtering
    model_knn = NearestNeighbors(
        metric='cosine', 
        algorithm='brute', 
        n_neighbors=20, 
        n_jobs=-1
    )
    model_knn.fit(sparse_matrix)
    return model_knn

# --- 5. RECOMMENDATION GENERATION ---
def get_recommendations(user_id, model_knn, sparse_matrix, user_movie_matrix, n_recs=10):
    """
    Finds the k-Nearest Neighbors for the user and generates recommendations.
    
    Args:
        user_id (int): The ID of the target user (1-based index).
        n_recs (int): The number of recommendations to return.
    """
    
    # Map user_id (1-based) to the row index (0-based) in the matrix
    try:
        user_row_index = user_movie_matrix.index.get_loc(user_id)
    except KeyError:
        return [f"User ID {user_id} not found. Please use an ID between 1 and {len(user_movie_matrix.index)}."]

    # Query the model for the nearest neighbors (users)
    distances, indices = model_knn.kneighbors(
        sparse_matrix[user_row_index], 
        n_neighbors=model_knn.n_neighbors + 1 # +1 to exclude the user itself
    )

    # Get the user's data and the data of their neighbors
    user_ratings = user_movie_matrix.iloc[user_row_index]
    neighbor_indices = indices.flatten()[1:] # Exclude the user itself (first index)
    neighbor_distances = distances.flatten()[1:]
    
    # Store recommendations: {movie_id: predicted_score}
    recommendation_scores = {}
    
    # Iterate through the neighbors
    for i, neighbor_index in enumerate(neighbor_indices):
        # Weight is 1 - distance (to get similarity)
        weight = 1 - neighbor_distances[i]
        
        neighbor_ratings = user_movie_matrix.iloc[neighbor_index]
        
        # Find movies the neighbor rated highly and the target user hasn't seen (rating == 0)
        unseen_movies = user_ratings[user_ratings == 0].index
        
        for movie_id in unseen_movies:
            neighbor_score = neighbor_ratings.loc[movie_id]
            
            # If the neighbor rated the movie (score > 0)
            if neighbor_score > 0:
                # Accumulate weighted score (a simplified prediction approach)
                current_score = recommendation_scores.get(movie_id, 0)
                recommendation_scores[movie_id] = current_score + (neighbor_score * weight)

    # Sort the recommendations by score
    sorted_recs = sorted(
        recommendation_scores.items(), 
        key=lambda item: item[1], 
        reverse=True
    )
    
    # Get the top N recommendations (movie IDs)
    top_n_movie_ids = [movie_id for movie_id, score in sorted_recs[:n_recs]]
    
    # --- BONUS: Get movie titles for a better output ---
    # Load movie titles (u.item) if available, otherwise just return IDs
    # For simplicity, we'll return the Movie ID. In a full system, you'd merge titles here.
    
    return [f"Movie ID {mid}" for mid in top_n_movie_ids]

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    sparse_matrix, user_movie_matrix = load_data()
    if sparse_matrix is not None:
        knn_model = train_model(sparse_matrix)
        
        # Test the system with a sample user (User ID 1)
        test_user_id = 1
        recommendations = get_recommendations(test_user_id, knn_model, sparse_matrix, user_movie_matrix, n_recs=5)
        
        print(f"\n--- Proactive Recommendations for User ID {test_user_id} ---")
        for i, rec in enumerate(recommendations):
            print(f"{i+1}. {rec}")
