from flask import Flask, render_template, request
from recommender_model import load_data, train_model, get_recommendations

# Initialize Flask app
app = Flask(__name__)

# Global variables to hold the model and data (loaded once at startup)
KNN_MODEL = None
SPARSE_MATRIX = None
USER_MOVIE_MATRIX = None
MAX_USER_ID = None

# --- Application Startup ---
def initialize_model():
    """Load data and train the model."""
    global KNN_MODEL, SPARSE_MATRIX, USER_MOVIE_MATRIX, MAX_USER_ID
    print("Loading data and training k-NN model...")
    SPARSE_MATRIX, USER_MOVIE_MATRIX = load_data()
    
    if SPARSE_MATRIX is not None:
        KNN_MODEL = train_model(SPARSE_MATRIX)
        MAX_USER_ID = len(USER_MOVIE_MATRIX.index)
        print("Model loaded and ready.")
    else:
        print("Model initialization failed. Check data file.")

# Load the model when the application starts
initialize_model()

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles the main interface for recommendations."""
    recommendations = None
    user_id = None
    error_message = None

    if request.method == 'POST':
        try:
            # Get user ID from the web form
            user_id = int(request.form['user_id'])
            
            if KNN_MODEL is None:
                error_message = "Recommendation system is not initialized. Check server logs."
            elif user_id < 1 or user_id > MAX_USER_ID:
                error_message = f"Invalid User ID. Please enter an ID between 1 and {MAX_USER_ID}."
            else:
                # Generate recommendations using the imported function
                recommendations = get_recommendations(
                    user_id, 
                    KNN_MODEL, 
                    SPARSE_MATRIX, 
                    USER_MOVIE_MATRIX, 
                    n_recs=10
                )
                
        except ValueError:
            error_message = "Please enter a valid number for the User ID."
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"

    # Render the HTML template with the results
    return render_template(
        'index.html', 
        recommendations=recommendations, 
        user_id=user_id, 
        error=error_message
    )

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
