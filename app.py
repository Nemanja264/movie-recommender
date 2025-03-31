from flask import Flask, render_template, jsonify, request
import pandas as pd
from movie_recommender import MovieRecommender
from requests import get
import json

app = Flask(__name__)

movie_df = pd.read_csv("movies.csv")

with open("genres", "r") as f:
    genres_map = json.load(f)

recommender = MovieRecommender(movie_df, genres_map)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend_movie/<title>')
def recommend_movies(title):
    recommended_movies = recommender.recommend(title, 10)

    return jsonify(recommended_movies)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5555, debug=True)