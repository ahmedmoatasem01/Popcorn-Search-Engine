# PopcornQuery-Search-Engine

Popcorn-Search-Engine is a powerful and interactive movie search engine built with **Python**, **Flask**, and **PyTerrier**, using **TMDb API** for real-time movie data. It supports multiple retrieval models and advanced features like semantic query expansion, watchlist management, and multimedia search.

---

## 🚀 Features

- 🔍 Full-text movie search (title, genre, director, actor, release year , poster , trailer)
- 🔥 Real-time **Trending Movies** from TMDb API
- 📈 Ranking models: BM25, TF-IDF, Hiemstra_LM, Semantic Search (SBERT)
- 🎞️ Real-time data from TMDb API
- 📺 "Your Watchlist" panel with movie descriptions, trailers, etc.
- 🗣️ Voice and 🖼️ image-based search input
- 🎨 Light/Dark theme toggle and background customization

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask, PyTerrier, SentenceTransformers
- **Frontend**: HTML, CSS, JavaScript
- **Data Sources**: TMDb API, custom Qrels

---

## 📦 Setup Instructions

1. Clone the repo :
   ```bash
   git clone https://github.com/ahmedmoatasem01/Popcorn-Search-Engine
   cd Popcorn-Search-Engine
   
2.Install the required libraries :
pip install flask
pip install pyterrier
pip install pandas
pip install requests
pip install sentence-transformers
pip install scikit-learn
pip install python-dotenv

3. TMDb API key :
TMDB_API_KEY=your_tmdb_api_key_here

4.Run the application :
python app.py

5.Open your browser and go to :
http://127.0.0.1:5000




