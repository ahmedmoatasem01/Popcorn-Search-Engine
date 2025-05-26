import requests
import pandas as pd
import os

API_KEY = 'API-KEY'
FILE_PATH = './Data/movie_qrels.csv'

def get_movie_data(query):
    if os.path.exists(FILE_PATH):
        df_existing = pd.read_csv(FILE_PATH)
        df_existing.columns = df_existing.columns.str.strip()
    else:
        df_existing = pd.DataFrame()

    data = []
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={query}"
    search_response = requests.get(search_url).json()

    for movie in search_response.get('results', []):
        title = movie['title']

        # Avoid adding duplicates by title
        if 'Title' in df_existing.columns and title in df_existing['Title'].values:
            continue

        movie_id = movie['id']
        details_response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}").json()
        credits_response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={API_KEY}").json()
        videos_response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={API_KEY}").json()

        release_date = movie.get('release_date', 'Unknown')
        poster_url = f"https://image.tmdb.org/t/p/w500{movie.get('poster_path', '')}"
        description = details_response.get('overview', 'No description available')
        genres = ', '.join([genre['name'] for genre in details_response.get('genres', [])])
        director = next((p['name'] for p in credits_response.get('crew', []) if p['job'] == 'Director'), 'Unknown')
        crew = ', '.join([p['name'] for p in credits_response.get('crew', [])[:5]])

        trailer = next((f"https://www.youtube.com/watch?v={vid['key']}"
                        for vid in videos_response.get('results', []) if vid['type'] == 'Trailer'), 'No trailer')

        data.append({
            "Title": title,
            "Release Date": release_date,
            "Poster URL": poster_url,
            "Description": description,
            "Genres": genres,
            "Director": director,
            "Crew": crew,
            "Trailer": trailer
        })

    df_new = pd.DataFrame(data)

    # Ensure Movie_ID exists in old data
    if 'Movie_ID' not in df_existing.columns:
        df_existing['Movie_ID'] = range(1, len(df_existing) + 1)

    # Assign Movie_IDs to new rows only
    start_id = df_existing['Movie_ID'].max() if not df_existing.empty else 0
    df_new['Movie_ID'] = range(start_id + 1, start_id + 1 + len(df_new))

    # Combine and save
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(FILE_PATH, index=False)

    return df_combined
