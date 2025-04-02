import time
from requests import get
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = "1ddbd50c511eba6d0d83d2b19a5b77b3"
BASE_URL = f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}&language=en-US&sort_by=popularity.desc"
GENRE_URL = f"https://api.themoviedb.org/3/genre/movie/list?api_key={API_KEY}&language=en-US"

def load_imdb_id(tmdb_id):
    IMDB_URL = f"https://api.themoviedb.org/3/movie/{tmdb_id}/external_ids?api_key={API_KEY}"

    try:
        response = get(IMDB_URL)
        data = response.json()

        return data.get("imdb_id")
    except Exception as e:
        print(f"Error fetching IMDb ID for TMDb ID {tmdb_id}: {e}")
        return None

def attach_imdb_links(movies):
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(load_imdb_id, movie["id"]): movie for movie in movies}

        for future in as_completed(futures):
            movie = futures[future]
            imdb_id = future.result()
            if imdb_id:
                movie['imdb_url'] = f"https://www.imdb.com/title/{imdb_id}/"
            else:
                movie['imdb_url'] = None

    return movies

def load_movies(URL = BASE_URL):
    all_movies = []

    for page in range(1,500):
        MOVIES_URL = f"{BASE_URL}&page={page}"

        try:
            response = get(MOVIES_URL)
            data = response.json()
            page_movies = data.get("results", [])
        except Exception as e:
            print(f"Failed to fetch page {page}: {e}")
            continue

        all_movies.extend(attach_imdb_links(page_movies))
    
        time.sleep(0.25)
    
    return all_movies

def load_genres(URL = GENRE_URL):
    genres_json = get(URL).json()
    genres_map = {genre['id']: genre['name'] for genre in genres_json['genres']}
    
    return genres_map

movies = pd.DataFrame(load_movies())
movies.to_csv("movies.csv", index=False)
'''
genres_map = load_genres()
with open("genres", "w") as f:
    json.dump(genres_map, f)'''
