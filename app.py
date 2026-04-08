from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Create user-item matrix
user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# Compute similarity
similarity = cosine_similarity(user_movie_matrix)
similarity_df = pd.DataFrame(similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# Recommendation function
def recommend_movies(user_id, top_n=3):
    # Check if user_id exists
    if user_id not in similarity_df.index:
        return None  # Return None if user doesn't exist
    
    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:]
    
    recommended_movies = {}
    
    for sim_user, score in similar_users.items():
        user_ratings = user_movie_matrix.loc[sim_user]
        
        for movie_id, rating in user_ratings.items():
            if rating > 0 and user_movie_matrix.loc[user_id][movie_id] == 0:
                recommended_movies[movie_id] = recommended_movies.get(movie_id, 0) + score * rating
    
    sorted_movies = sorted(recommended_movies.items(), key=lambda x: x[1], reverse=True)
    
    movie_titles = [movies[movies['movieId'] == mid]['title'].values[0] for mid, _ in sorted_movies[:top_n]]
    
    return movie_titles

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    error = None
    
    if request.method == 'POST':
        try:
            user_id = int(request.form['user_id'])
            result = recommend_movies(user_id)
            
            if result is None:
                error = f"User ID {user_id} not found. Valid range: 1-{int(similarity_df.index.max())}"
            else:
                recommendations = result
        except ValueError:
            error = "Please enter a valid user ID (must be a number)"
        except Exception as e:
            error = f"An error occurred: {str(e)}"
    
    return render_template('index.html', recommendations=recommendations, error=error)

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Open your browser and go to http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)