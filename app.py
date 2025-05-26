from PIL import Image
import imagehash
from io import BytesIO
import re
from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask import Flask, send_from_directory
from Backend.API import get_movie_data
from models.retreivers import (
    rebuild_corpus_and_index,
    search_TFIDF,
    search_bm25,
    search_unigram,
    search_bm25_expansion_minilm
)
import json
import pandas as pd
import requests
#import webbrowser
#import threading

app = Flask(__name__)




df = pd.read_csv("Data/movie_qrels.csv")
movie_titles = df["Title"].dropna().unique().tolist()

@app.route("/api/autocomplete", methods=["GET"])
def autocomplete():
    query = request.args.get("query", "").lower()
    suggestions = [title for title in movie_titles if title.lower().startswith(query)][:10]
    return jsonify({"suggestions": suggestions})



@app.route('/api/image-search', methods=['POST'])
def image_search():
    if 'image' not in request.files:
        return jsonify([]), 400

    uploaded_file = request.files['image']
    try:
        query_img = Image.open(uploaded_file)
        query_hash = imagehash.phash(query_img)
    except Exception as e:
        return jsonify({"error": f"Could not process image: {str(e)}"}), 500

    matches = []
    for _, row in df.iterrows():
        poster_url = row.get("Poster URL")
        if poster_url and poster_url.startswith("http"):
            try:
                response = requests.get(poster_url, timeout=5)
                poster_img = Image.open(BytesIO(response.content))
                poster_hash = imagehash.phash(poster_img)
                diff = query_hash - poster_hash

                # You can tune this threshold for similarity
                if diff < 10:
                    matches.append({
                        "title": row.get("Title", ""),
                        "poster": poster_url,
                        "text": row.get("Description", ""),
                        "trailer": row.get("Trailer", "")
                    })
            except Exception as e:
                continue

    return jsonify(matches[:15])  


@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory('images', filename)


@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username")
    password = request.form.get("password")

    if username == "admin" and password == "admin":
        return redirect(url_for("admin_page"))
    elif username == "user" and password == "user":
        return redirect(url_for("search_page"))
    else:
        return render_template("index.html", error="Invalid login")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/admin')
def admin_page():
    return render_template('admin.html')

@app.route('/search')
def search_page():
    trending = fetch_trending_movies()
    return render_template("search.html", trending_movies=trending)

@app.route('/advancedsearch')
def advanced_search():
    return render_template('advancedsearch.html')


CSV_PATH = "movie_qrels.csv"

@app.route("/admin/upload", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return jsonify({"message": "No file uploaded."}), 400

    file = request.files["file"]
    try:
        df4 = pd.read_csv(file)
        df4.to_csv(CSV_PATH, index=False) 
#        process_and_index_file(save_path)
#       return jsonify({"message": f"File '{file.filename}' uploaded and indexed successfully!"})
        return jsonify({"message": "CSV uploaded and saved successfully!"})
    except Exception as e:
        return jsonify({"message": f" Error reading CSV: {str(e)}"}), 500
    




@app.route("/admin/fetch", methods=["POST"])
def fetch_movie():
    data = request.json
    query = data.get("query")
    if not query:
        return jsonify({"message": "No movie title provided."}), 400

    try:
        get_movie_data(query)
        rebuild_corpus_and_index()
        return jsonify({"message": f"Movie '{query}' fetched and indexed!"})
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500






API_KEY = 'API-KEY'

def fetch_trending_movies():
    TMDB_TRENDING_URL = f"https://api.themoviedb.org/3/trending/movie/week?api_key={API_KEY}"
    TMDB_VIDEO_URL = "https://api.themoviedb.org/3/movie/{}/videos?api_key={}"

    response = requests.get(TMDB_TRENDING_URL).json()
    trending_movies = []

    for movie in response.get("results", []):
        title = movie.get("title")
        poster_path = movie.get("poster_path")
        movie_id = movie.get("id")

        trailer_url = ""
        video_data = requests.get(TMDB_VIDEO_URL.format(movie_id, API_KEY)).json()
        for vid in video_data.get("results", []):
            if vid["type"] == "Trailer" and vid["site"] == "YouTube":
                trailer_url = f"https://www.youtube.com/watch?v={vid['key']}"
                break

        trending_movies.append({
            "title": title,
            "poster_url": f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "N/A",
            "trailer": trailer_url or "#"
        })

    return trending_movies



def extract_fields(text):
    pattern = r'(Pro_Title|Pro_Description|Pro_Genres|Pro_Director|Pro_Crew|Release Date):\s*(.*?)(?=\b(?:Pro_Title|Pro_Description|Pro_Genres|Pro_Director|Pro_Crew|Release Date):|$)'
    matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return {k.strip(): v.strip() for k, v in matches}



def clean_text(text):
    return re.sub(
        r'\b(?:Pro_Title|Pro_Description|Pro_Genres|Pro_Director|Pro_Crew|Movie_ID|Release Date)\s*:', 
        '', 
        text, 
        flags=re.IGNORECASE
    ).strip()



@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.get_json()
    query = data.get("query")
    method = data.get("method", "tfidf")

    #     # Fetch movie data from TMDb API and update CSV if new
    #     get_movie_data(query)
    #reindexing or delete indeer and create new one when searching

    if method == "tfidf":
        results = search_TFIDF(query)
    elif method == "bm25":
        results = search_bm25(query)
    elif method == "unigram":
        results = search_unigram(query)
    elif method == "semantic":
        results = search_bm25_expansion_minilm(query)
    else:
        return jsonify([{"title": "Error", "text": "Invalid method", "poster": "", "trailer": ""}])

    for item in results:
        fields = extract_fields(item.get("text", ""))
        item["title"] = fields.get("Pro_Title", item.get("title", "Untitled"))
        item["text"] = fields.get("Pro_Description", "")
        item["genres"] = fields.get("Pro_Genres", "N/A")
        item["director"] = fields.get("Pro_Director", "N/A")
        item["crew"] = fields.get("Pro_Crew", "N/A")
        item["release_year"] = fields.get("Release Date", "N/A")[:4]  


    return jsonify(results)



@app.route("/api/advanced_search", methods=["POST"])
def advanced_search_api():
    data = request.get_json()
    title = data.get("title", "").lower()
    year = data.get("year")
    genre = data.get("genre", "").lower()
    director = data.get("director", "").lower()
    actor = data.get("actor", "").lower()

    df = pd.read_csv("Data/movie_qrels.csv")

    def parse_genres(genre_str):
        try:
            genre_str = genre_str.replace("'", '"')
            genres_list = json.loads(genre_str)
            return [g['name'].lower() for g in genres_list]
        except:
            return []

    df['genres_list'] = df['Genres'].apply(parse_genres)
    df['release_year'] = pd.to_datetime(df['Release Date'], errors='coerce').dt.year

    results = df.copy()
    if title:
        results = results[results['Title'].str.lower().str.contains(title, na=False)]
    if year and year.isdigit():
        results = results[results['release_year'] == int(year)]
    if genre:
        results = results[results['genres_list'].apply(lambda g: genre in g)]
    if director:
        results = results[results['Director'].str.lower().str.contains(director, na=False)]
    if actor:
        results = results[results['Crew'].str.lower().str.contains(actor, na=False)]

    results = results.head(15).fillna("")
    return jsonify(results.to_dict(orient="records"))



if __name__ == '__main__':
    app.run(debug=True)
