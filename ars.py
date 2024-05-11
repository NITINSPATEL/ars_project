import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score
from surprise.accuracy import rmse

# Parse movies.csv
movies_df = pd.read_csv('movies.csv')

# Create content-based recommendation system
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['genres'].fillna(''))
content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Parse ratings.csv
ratings_df = pd.read_csv('ratings.csv')

# Create collaborative filtering recommendation system
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)


# Generate recommendations using content-based system
def content_based_recommendation(user_movies):
    movie_indices = [movies_df.index[movies_df['movieId'] == movie_id].tolist()[0] for movie_id in user_movies]
    avg_similarity = content_similarity[movie_indices].mean(axis=0)
    top_indices = avg_similarity.argsort()[::-1][:10]
    return movies_df.iloc[top_indices]['movieId'].tolist()

# Generate recommendations using collaborative filtering system
def collaborative_filtering_recommendation(user_id):
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    user_not_rated = movies_df[~movies_df['movieId'].isin(user_ratings['movieId'])]['movieId']
    predictions = [algo.predict(user_id, movie_id).est for movie_id in user_not_rated]
    top_indices = sorted(range(len(predictions)), key=lambda i: predictions[i], reverse=True)[:10]
    return [user_not_rated.iloc[i] for i in top_indices]


# Metrics for collaborative filtering system
def evaluate_collaborative_filtering(predictions):
    return rmse(predictions)

# Metrics for content-based recommendation system
def evaluate_content_based(user_movies, recommended_movies):
    true_positives = len(set(user_movies) & set(recommended_movies))
    precision = true_positives / len(recommended_movies) if len(recommended_movies) > 0 else 0
    recall = true_positives / len(user_movies) if len(user_movies) > 0 else 0
    return precision, recall

# Compare and contrast the two systems
def compare_systems(user_movies, collaborative_predictions, content_based_recommendations):
    collab_rmse = evaluate_collaborative_filtering(collaborative_predictions)
    content_precision, content_recall = evaluate_content_based(user_movies, content_based_recommendations)

    print("Collaborative Filtering RMSE:", collab_rmse)
    print("Content-Based Precision:", content_precision)
    print("Content-Based Recall:", content_recall)

#example
user_id = 1
user_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
collaborative_predictions = algo.test(trainset.build_testset())
content_based_recommendations = content_based_recommendation(user_movies)

compare_systems(user_movies, collaborative_predictions, content_based_recommendations)
