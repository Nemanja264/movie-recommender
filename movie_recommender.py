import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
import heapq
import ast
from db import get_movie_info, get_similar_movies, get_similar_movies_info

class MovieRecommender:
    def __init__(self, df, genres_map):
        self.df = df.copy()
        self.genres_map = genres_map
        self.similarity = load_npz("similarity_matrix_tuned200.npz")
        self.prepare_data()
        #self.build_similarity_matrix(self.df['combined'])

    def prepare_data(self):
        self.df['genre_ids'] = self.df['genre_ids'].apply(ast.literal_eval)
        self.df['genres'] = self.df['genre_ids'].apply(self.convert_genres)
        
        self.df = self.df[['id','title', 'release_date', 'genres', 'imdb_url', 'overview']]
        self.df = self.df.drop_duplicates(subset=["title", "release_date"]).reset_index(drop=True)

        self.df['release_year'] = pd.to_datetime(self.df["release_date"], errors="coerce").dt.year
        self.df['release_year'] = self.df['release_year'].fillna(1900).astype(int)

        self.df['overview'] = self.df['overview'].fillna("").astype(str)

        self.df = self.df.drop(columns="release_date")

        self.combine_columns('title', 'release_year', 'genres', 'overview')

    def find_movie(self, movie_title):
        matches = self.df[self.df['title'] == movie_title]

        if matches.empty:
            matches = self.df[self.df['title'].str.contains(movie_title, case=False, na=False)]

        if matches.empty:
            print(f"Matches for {movie_title} were not found")
            return None
        
        return matches.index[0]
        
    def recommend(self, movie_title, top_n = 5):
        try:
            ind = self.find_movie(movie_title)
            if ind is None:
                return None
           
            similar_movies = get_similar_movies(int(self.df.iloc[ind].id))[:top_n]

            movie_ids = [i for i,_ in similar_movies]
            
            return get_similar_movies_info(movie_ids)
        
        except Exception as e:
             print(f"Error in recommend(): {e}")
             return [{"error": str(e)}]


    def convert_genres(self, genre_ids):
        return ' '.join([self.genres_map.get(str(gid), 'Unknown') for gid in genre_ids])
    
    def get_reduced_sim_matrix(self, sim_matrix, top_n = 100):
        reduced_sim_matrix = np.zeros_like(sim_matrix)

        for i, row in enumerate(sim_matrix):
            top_indices_scores = heapq.nlargest(top_n, enumerate(row), key=lambda x: x[1])

            top_indices, top_scores = zip(*top_indices_scores)

            reduced_sim_matrix[i, list(top_indices)] = top_scores

        return reduced_sim_matrix
    
    def get_sparse_sim_matrix(self, sim_matrix, top_n=100):
        reduced_sim_matrix = self.get_reduced_sim_matrix(sim_matrix, top_n=top_n)

        sparse_matrix = csr_matrix(reduced_sim_matrix)

        return sparse_matrix
    
    def build_similarity_matrix(self, column, batch_size=32): # load it once(in file) no need to load it everytime
        model = SentenceTransformer('movie_rec200')

        embeddings = [
                embedding 
                for i in range(0, int(len(column)), batch_size)
                for embedding in model.encode(column.iloc[i:i+batch_size].tolist(), show_progress_bar=True)
            ]
        
        sim_matrix = cosine_similarity(np.array(embeddings))
        self.similarity = self.get_sparse_sim_matrix(sim_matrix)
        
        save_npz("similarity_matrix_tuned200.npz", self.similarity)
    

    def combine_columns(self, *column_names):
        self.df['combined'] = ''
        for column in column_names:
            self.df['combined'] += self.df[column].astype(str) + " "
