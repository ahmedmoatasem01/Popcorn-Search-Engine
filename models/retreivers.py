from utils.preprocessing import processText
import pandas as pd
import pyterrier as pt
import re
import requests
import os

if not pt.started():
    pt.init()
#reading API 
# df = pd.read_csv("Data/movie_qrels.csv")

# df["Pro_Title"] = df["Title"].apply(processText)
# df["Pro_Description"] = df["Description"].apply(processText)
# df["Pro_Genres"] = df["Genres"].astype(str).apply(processText)
# df["Pro_Director"] = df["Director"].apply(processText)
# df["Pro_Crew"] = df["Crew"].astype(str).apply(processText)
# df["P_Trailer"]=df["Trailer"]
# df["Poster"]=df["Poster URL"]

# df.drop(columns=['Genres', 'Director', 'Crew','Description','Title','Trailer','Poster URL'], inplace=True, errors='ignore')

# #reading dataset
# df2 = pd.read_csv("Data/scrapedmove_dataset.csv")

# df2.drop(columns=['procced_genres'],inplace=True)
# df2.drop(columns=['title'],inplace=True)

def extract_names(text):
     matches = re.findall(r"'name':\s*'([^']+)'", text)
     return ", ".join(matches)
    
# df2["Pre_Title"]=df2["original_title"].apply(processText)
# df2["Pre_genres"]=df2["genres"].apply(extract_names)
# df2["Proceed_Genre"]=df2["Pre_genres"].apply(processText)   

# df2.drop(columns=['genres', 'original_language', 'original_title','video','id_p','Pre_genres'], inplace=True, errors='ignore')


# ########################################################

# #create corpus
# with open("corpus.txt", "w") as filee:

#     df[df.columns] = df[df.columns].astype(str)
#     df2[df2.columns] = df2[df2.columns].astype(str)



#     for index, row in df.iterrows():
#         filee.write(f"docid {index + 1}:\n")
#         for col in df.columns:

#             filee.write(f"\t{col}:{row[col]}\t\n")
#         filee.write("\n############################################\n\n")

#     for index, row in df2.iterrows():
#         filee.write(f"docid {index + 1}:\n")
#         for col in df2.columns:
           
#           filee.write(f"\t{col}:{row[col]}\t\n")
#         filee.write("\n############################################\n\n")

# #########################################################################################

# #open the corpus

# corpus_file = "corpus.txt"

# documents = []
# doc_id = 0

# with open(corpus_file, "r", encoding="utf-8") as f:
#     doc_text = ""

#     for line in f:
#         line = line.strip()

#         if line.startswith("docid"):
#             if doc_text:
#                 documents.append({"docno": str(doc_id), "text": doc_text})
#                 doc_id += 1
#             doc_text = ""
#         elif line and not line.startswith("############################################"):
#             doc_text += " " + line

#     if doc_text:
#         documents.append({"docno": str(doc_id), "text": doc_text})

# df3 = pd.DataFrame(documents)

# ####################################################

# #indexing

# index_path = os.path.abspath("text_index")
# os.makedirs(index_path, exist_ok=True)


# # Only index if it doesn't exist
# if not os.path.exists(os.path.join(index_path, "data.properties")):
#     indexer = pt.IterDictIndexer(index_path, fields=["text"], meta=["docno"])
#     print("Indexing documents...")
#     index_ref = indexer.index(df3.to_dict("records"))
# else:
#     print("Index exists. Skipping indexing.")
#     index_ref = index_path

# # Load the index
# index = pt.IndexFactory.of(index_ref)

def rebuild_corpus_and_index():
    global df, df2, df3, index

    # Re-load datasets
    df = pd.read_csv("Data/movie_qrels.csv")
    df2 = pd.read_csv("Data/scrapedmove_dataset.csv")

    # Preprocess df
    df["Pro_Title"] = df["Title"].apply(processText)
    df["Pro_Description"] = df["Description"].apply(processText)
    df["Pro_Genres"] = df["Genres"].astype(str).apply(processText)
    df["Pro_Director"] = df["Director"].apply(processText)
    df["Pro_Crew"] = df["Crew"].astype(str).apply(processText)
    df["P_Trailer"] = df["Trailer"]
    df["Poster"] = df["Poster URL"]
    df.drop(columns=['Genres', 'Director', 'Crew', 'Description', 'Title', 'Trailer', 'Poster URL'], inplace=True, errors='ignore')

    # Preprocess df2
    df2.drop(columns=['procced_genres', 'title'], inplace=True, errors='ignore')
    df2["Pre_Title"] = df2["original_title"].apply(processText)
    df2["Pre_genres"] = df2["genres"].apply(extract_names)
    df2["Proceed_Genre"] = df2["Pre_genres"].apply(processText)
    df2.drop(columns=['genres', 'original_language', 'original_title', 'video', 'id_p', 'Pre_genres'], inplace=True, errors='ignore')

    # Build corpus
    with open("corpus.txt", "w") as filee:
        df[df.columns] = df[df.columns].astype(str)
        df2[df2.columns] = df2[df2.columns].astype(str)

        for index, row in df.iterrows():
            filee.write(f"docid {index + 1}:\n")
            for col in df.columns:
                filee.write(f"\t{col}:{row[col]}\t\n")
            filee.write("\n############################################\n\n")

        for index, row in df2.iterrows():
            filee.write(f"docid {index + 1}:\n")
            for col in df2.columns:
                filee.write(f"\t{col}:{row[col]}\t\n")
            filee.write("\n############################################\n\n")

    # Rebuild df3
    documents = []
    doc_id = 0
    with open("corpus.txt", "r", encoding="utf-8") as f:
        doc_text = ""
        for line in f:
            line = line.strip()
            if line.startswith("docid"):
                if doc_text:
                    documents.append({"docno": str(doc_id), "text": doc_text})
                    doc_id += 1
                doc_text = ""
            elif line and not line.startswith("############################################"):
                doc_text += " " + line
        if doc_text:
            documents.append({"docno": str(doc_id), "text": doc_text})
    df3 = pd.DataFrame(documents)

    # Delete old index and create new one
    index_path = os.path.abspath("text_index")
    if os.path.exists(index_path):
        import shutil
        shutil.rmtree(index_path)
    os.makedirs(index_path, exist_ok=True)

    indexer = pt.IterDictIndexer(index_path, fields=["text"], meta=["docno"])
    index_ref = indexer.index(df3.to_dict("records"))
    index = pt.IndexFactory.of(index_ref)


#ranking




def extract_fields(text):
    fields = {
        "Pro_Title": "Untitled",
        "Pro_Description": "No description available.",
        "Pro_Genres": "N/A",
        "Pro_Director": "N/A",
        "Pro_Crew": "N/A",
        "Release Date": "N/A"
    }

    pattern = r'(\bPro_Title|\bPro_Description|\bPro_Genres|\bPro_Director|\bPro_Crew|\bRelease Date):\s*(.*?)\t+'
    matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)

    for key, value in matches:
        fields[key.strip()] = value.strip()

    return fields




def search_TFIDF(query):
    query = processText(query)
    tfidf = pt.BatchRetrieve(index, wmodel="TF_IDF", num_results=100)
    tfidf_results = tfidf.search(query)

    results = []
    for _, row in tfidf_results.iterrows():
        docno = row["docno"]
        try:
            doc_text = df3[df3["docno"] == docno]["text"].values[0]
            fields = extract_fields(doc_text)
            results.append({
                "title": fields.get("Pro_Title", "Untitled"),
                "text": fields.get("Pro_Description", "No description available."),
                "genres": fields.get("Pro_Genres", "N/A"),
                "director": fields.get("Pro_Director", "N/A"),
                "crew": fields.get("Pro_Crew", "N/A"),
                "release_year": fields.get("Release Date", "N/A")[:4],
                "poster": df.iloc[int(docno)].get("Poster", "N/A") if int(docno) < len(df) else "N/A",
                "trailer": df.iloc[int(docno)].get("P_Trailer", "") if int(docno) < len(df) else ""
            })
        except:
            results.append({
                "title": "Error",
                "text": "Not found",
                "genres": "N/A",
                "director": "N/A",
                "crew": "N/A",
                "release_year": "N/A",
                "poster": "N/A",
                "trailer": "N/A"
            })
    return results


def search_bm25(query):
    query = processText(query)
    bm25_retr = pt.BatchRetrieve(index, wmodel="BM25", num_results=100)
    bm25_results = bm25_retr.search(query)

    results = []

    for _, row in bm25_results.iterrows():
        docno = row["docno"]
        try:
            doc_text = df3[df3["docno"] == docno]["text"].values[0]
            fields = extract_fields(doc_text)
            results.append({
                "title": fields.get("Pro_Title", "Untitled"),
                "text": fields.get("Pro_Description", "No description available."),
                "genres": fields.get("Pro_Genres", "N/A"),
                "director": fields.get("Pro_Director", "N/A"),
                "crew": fields.get("Pro_Crew", "N/A"),
                "release_year": fields.get("Release Date", "N/A")[:4],
                "poster": df.iloc[int(docno)].get("Poster", "N/A") if int(docno) < len(df) else "N/A",
                "trailer": df.iloc[int(docno)].get("P_Trailer", "") if int(docno) < len(df) else ""
            })
        except Exception:
            results.append({
                "title": "Error",
                "text": "Not found",
                "genres": "N/A",
                "director": "N/A",
                "crew": "N/A",
                "release_year": "N/A",
                "poster": "N/A",
                "trailer": "N/A"
            })

    return results





def search_unigram(query):
    query = processText(query)
    unigram = pt.BatchRetrieve(index, wmodel="Hiemstra_LM", num_results=100)
    unigram_results = unigram.search(query)

    results = []
    for _, row in unigram_results.iterrows():
        docno = row["docno"]
        try:
            doc_text = df3[df3["docno"] == docno]["text"].values[0]
            fields = extract_fields(doc_text)
            results.append({
                "title": fields.get("Pro_Title", "Untitled"),
                "text": fields.get("Pro_Description", "No description available."),
                "genres": fields.get("Pro_Genres", "N/A"),
                "director": fields.get("Pro_Director", "N/A"),
                "crew": fields.get("Pro_Crew", "N/A"),
                "release_year": fields.get("Release Date", "N/A")[:4],
                "poster": df.iloc[int(docno)].get("Poster", "N/A") if int(docno) < len(df) else "N/A",
                "trailer": df.iloc[int(docno)].get("P_Trailer", "") if int(docno) < len(df) else ""
            })
        except:
            results.append({
                "title": "Error",
                "text": "Not found",
                "genres": "N/A",
                "director": "N/A",
                "crew": "N/A",
                "release_year": "N/A",
                "poster": "N/A",
                "trailer": "N/A"
            })
    return results






#Query expansion


def extract_terms_from_docs(doc_ids, df3, processText):
    all_terms = []
    for doc_id in doc_ids:
        try:
            text = df3[df3['docno'] == doc_id]['text'].values[0]
            terms = processText(text)
            all_terms.extend(terms.split())
        except IndexError:
            continue
    return all_terms


from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_semantic_expansion(query_terms, feedback_terms):
    query_embedding = model.encode(" ".join(query_terms), convert_to_tensor=True)
    unique_terms = list(set(feedback_terms) - set(query_terms))

    if not unique_terms:
        return []

    term_embeddings = model.encode(unique_terms, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, term_embeddings)[0]
    top_indices = cos_scores.argsort(descending=True)[:5]
    return [unique_terms[i] for i in top_indices]




def search_bm25_expansion_minilm(query):
    search_results = []
    q_processed = processText(query).split()

    # Step 1: Initial retrieval with BM25
    bm25_retr = pt.BatchRetrieve(index, wmodel="BM25", num_results=100)
    initial_results = bm25_retr.search(" ".join(q_processed))
    top_docs = initial_results.head(5)['docno'].tolist()

    feedback_terms = extract_terms_from_docs(top_docs, df3, processText)
  
    semantic_expansion = get_semantic_expansion(q_processed, feedback_terms)

    expanded_query = " ".join(q_processed * 3 + semantic_expansion)

    expanded_results = bm25_retr.search(expanded_query)

    query_embedding = model.encode(" ".join(q_processed), convert_to_tensor=True)

    for _, row in expanded_results.iterrows():
        docno = row['docno']
        try:
            doc_text = df3[df3['docno'] == docno]['text'].values[0]
            doc_embedding = model.encode(doc_text, convert_to_tensor=True)
            sim_score = util.pytorch_cos_sim(query_embedding, doc_embedding).item()

            movie_idx = int(docno)
            movie_row = df.iloc[movie_idx]

            search_results.append({
                "doc_id": docno,
                "score": sim_score,
                "title": movie_row.get("Pro_Title", "Unknown"),
                "poster": movie_row.get("Poster", "No poster"),
                "trailer": movie_row.get("P_Trailer", "No trailer"),
                "text": doc_text[:300] + "..."
            })
        except:
            search_results.append({
                "doc_id": docno,
                "score": 0,
                "title": "Error",
                "poster": "N/A",
                "trailer": "N/A",
                "text": "Not found"
            })

    # Step 8: Sort results by similarity score
    search_results.sort(key=lambda x: x['score'], reverse=True)
    return search_results

# def interactive_search():
#     import time
#     query = input(" Enter the movie name to search: ")

#     start_time = time.time()
#     results = search_bm25_expansion_minilm(query)
#     end_time = time.time()

#     print(f"\n Search Time: {end_time - start_time:.4f} seconds\n")

#     for r in results:
#         print(f"{r['title']}")
#         print(f" Poster: {r['poster']}")
#         print(f"▶ Trailer: {r['trailer']}")
#         print(f" Snippet: {r['text']}\n")


