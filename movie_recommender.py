import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class MovieRecommender:
    def __init__(self, df, genres_map):
        self.df = df.copy()
        self.genres_map = genres_map
        self.similarity = None #np.load("similarity_matrix.npy")
        self.prepare_data()
        self.build_similarity_matrix(self.df['combined'])

    def prepare_data(self):
        self.df['genres'] = self.df['genre_ids'].apply(self.convert_genres)

        self.df = self.df[['title', 'release_date', 'genres', 'imdb_url', 'overview']]
        self.df = self.df.drop_duplicates(subset=["title", "release_date"]).reset_index(drop=True)

        self.df['release_year'] = pd.to_datetime(self.df["release_date"], errors="coerce").dt.year
        self.df['release_year'] = self.df['release_year'].fillna(1900).astype(int)

        self.df = self.df.drop(columns="release_date")

        self.combine_columns('title', 'release_year', 'genres', 'overview')
    
    def recommend(self, movie_title, top_n = 5):
        try:
            matches = self.df[self.df['title'].str.contains(movie_title, case=False, na=False)]
            if matches.empty:
                print(f"Matches for {movie_title} were not found")
                return None

            ind = matches.index[0]

            sim_scores = list(enumerate(self.similarity[ind]))

            similar_movies = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

            print(f"\nTop {top_n} movies similar to '{self.df.iloc[ind].title}':\n")

            return [{
                    "title": f"{self.df.iloc[i].title}",
                    "release_year": f"{self.df.iloc[i].release_year}",
                    "imdb_link": f"{self.df.iloc[i].imdb_url}",
                    "similarity": float(round(score, 2))
                    }
                    for i, score in similar_movies
                ]
        
        except Exception as e:
             print(f"Error in recommend(): {e}")
             return [{"error": str(e)}]


    def convert_genres(self, genre_ids):
        return ' '.join([self.genres_map.get(gid, 'Unknown') for gid in genre_ids])
    
    def build_similarity_matrix(self, column): # load it once(in file) no need to load it everytime
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(column.tolist(), show_progress_bar=True)

        self.similarity = cosine_similarity(embeddings)
        #np.save("similarity_matrix.npy", self.similarity)
    
    def combine_columns(self, *column_names):
        self.df['combined'] = ''
        for column in column_names:
            self.df['combined'] += self.df[column].astype(str) + " "
