from flask import Flask, render_template, jsonify, request
import pandas as pd
from movie_recommender import MovieRecommender
from requests import get
import json

app = Flask(__name__)

movie_df = pd.read_csv("movies.csv")

with open("genres", "r") as f:
    genres_map = json.load(f)

recommender = MovieRecommender(movie_df.copy().iloc[:4000].reset_index(drop=True), genres_map)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend_movie/<title>/<numRecs>')
def recommend_movies(title, numRecs):
    recommended_movies = recommender.recommend(title, int(numRecs))

    return jsonify(recommended_movies)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5555, debug=True)
