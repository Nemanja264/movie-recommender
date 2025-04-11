from pymongo import MongoClient, ASCENDING
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

# Replace this with your connection string
connection_string = "mongodb+srv://nem2604:ieO9q3UhlOZwIvrW@cluster-movie.aqzobmz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster-movie"

movies_df = pd.read_csv("movies.csv")
movies_df = movies_df.drop_duplicates(subset=["title", "release_date"]).reset_index(drop=True)

movies_df['release_year'] = pd.to_datetime(movies_df["release_date"], errors="coerce").dt.year
movies_df['release_year'] = movies_df['release_year'].fillna(1900).astype(int)

# Connect to the cluster
client = MongoClient(connection_string)

# Get the database and collection
db = client['movie-recommender']  # Use your actual database name here
collection = db['similarity_matrix_tuned200']  # Example collection name
collection.create_index([('movie_id', ASCENDING)])

def fill_collection(collection, movies_df):
    similarity_matrix = load_npz("similarity_matrix_tuned200.npz")

    bulk_data = []
    # Example: Fetch all movie similarity data
    for i,_ in enumerate(similarity_matrix):
        movie = movies_df.iloc[i]  # Get movie details by index
        
        start_idx = similarity_matrix.indptr[i]
        end_idx = similarity_matrix.indptr[i + 1]
            
        movie_similarities = similarity_matrix.data[start_idx:end_idx]
        movie_indices = similarity_matrix.indices[start_idx:end_idx]

        sim_scores = list(zip(movie_indices, movie_similarities))
        similar_movies = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:101]

        similarity_data = {
            'movie_id': int(movie['id']),  # TMDB movie ID
            'title': str(movie['title']),
            'release_year': str(movie['release_year']),
            'imdb_link': str(movie['imdb_url']),
            'similar_movie_ids': [(int(movies_df.iloc[idx].id), float(score)) for idx, score in similar_movies] # Convert numpy array to list
        }

        bulk_data.append(similarity_data)

    if bulk_data:
        collection.insert_many(bulk_data)

    print("Similarity matrix with movie data uploaded to MongoDB!")
    row_count = collection.count_documents({})
    print(f"Number of rows: {row_count}/{len(movies_df)}")

def get_movie_title(movie_id):
    movie = collection.find_one({"movie_id": movie_id})

    if movie:
        return movie.get("title", "")
    else:
        print(f"Movie with id {movie_id} doesnt exist")
        return None
    
def get_movie_id(movie_title):
    movie = collection.find_one({"title": movie_title})

    if movie:
        return int(movie.get("movie_id", -1))
    else:
        print(f"Movie with id {movie_title} doesnt exist")
        return None

def get_imdb_link(movie_id):
    movie = collection.find_one({"movie_id": movie_id})

    if movie:
        return movie.get("imdb_link", "")
    else:
        print(f"Movie with id {movie_id} doesnt exist")
        return None
    
def get_release_year(movie_id):
    movie = collection.find_one({"movie_id": movie_id})

    if movie:
        return int(movie.get("release_year", -1))
    else:
        print(f"Movie with id {movie_id} doesnt exist")
        return None

def get_similar_movies(movie_id):
    movie = collection.find_one({"movie_id": movie_id}, {"similar_movie_ids": 1, "_id": 0})

    if movie:
        return movie.get("similar_movie_ids", [])
    else:
        print(f"Movie with id {movie_id} doesnt exist")
        return None
    
def get_movie_info(movie_id):
    movie = collection.find_one({"movie_id": movie_id}, {"title": 1, "release_year": 1, "imdb_link": 1, "_id": 0})

    if movie:
        return {
        "title": movie.get("title", ""),
        "release_year": int(movie.get("release_year", -1)),
        "imdb_link": movie.get("imdb_link", "")
    }
    else:
        print(f"Movie with id {movie_id} doesnt exist")
        return None
    
def get_similar_movies_info(movie_ids):
    similar_movies = list(collection.find({"movie_id": {"$in": movie_ids}}, 
                                {"movie_id": 1, "title": 1, "release_year": 1, "imdb_link": 1, "_id": 0}))
    
    return sorted(similar_movies, key=lambda x: movie_ids.index(x['movie_id']))

#fill_collection(collection, movies_df)
#print(f"{get_similar_movies(1125899)}")
