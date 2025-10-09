import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import nltk
nltk.download("punkt")
nltk.download("stopwords")

# --- Config ---
BOOKS_FILE = "books.csv"
AUTHOR_TEXTS_FILE = "author_texts.csv"   # nieuw: bestand voor auteurs-teksten
CHUNKED_TEXTS_FILE = "author_chunks.csv"

# --- Data laden ---
def load_books():
    if os.path.exists(BOOKS_FILE):
        df = pd.read_csv(BOOKS_FILE)
        if "Beschrijving" not in df.columns:
            df["Beschrijving"] = ""
        return df
    else:
        return pd.DataFrame(columns=["Titel", "Auteur", "Genre", "Beschrijving"])

books_df = load_books()

# --- Preprocessing boeken (beschrijvingen -> TF-IDF) ---
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = vectorizer.fit_transform(books_df["Beschrijving"].fillna(""))

# --- Split dataset ---
train_df, test_df = train_test_split(books_df, test_size=0.30, random_state=42)
valid_df, test_df = train_test_split(test_df, test_size=0.50, random_state=42)

print("📊 Dataset verdeling (boeken):")
print(f"Train: {len(train_df)} boeken")
print(f"Validatie: {len(valid_df)} boeken")
print(f"Test: {len(test_df)} boeken")

# --- Queryfunctie ---
def zoek_boeken(query, top_n=5):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:top_n]
    results = []
    for idx in top_indices:
        results.append({
            "Titel": books_df.iloc[idx]["Titel"],
            "Auteur": books_df.iloc[idx]["Auteur"],
            "Score": round(similarities[idx], 3)
        })
    return results

# ========================
# Auteursteksten verwerken
# ========================

def clean_text(text, lang="dutch"):
    """Schoon tekst op: lowercase, verwijder leestekens en stopwoorden."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Zà-ÿ\s]", "", text)  # verwijder cijfers/leestekens
    tokens = word_tokenize(text)
    stops = set(stopwords.words(lang))
    tokens = [t for t in tokens if t not in stops]
    return " ".join(tokens)

def chunk_text(text, chunk_size=300):
    """Splits tekst in chunks van ~300 woorden."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def preprocess_author_texts():
    if not os.path.exists(AUTHOR_TEXTS_FILE):
        print("⚠️ Geen auteursteksten gevonden.")
        return None
    
    df = pd.read_csv(AUTHOR_TEXTS_FILE)  # verwacht kolommen: Auteur, Tekst
    all_chunks = []
    
    for _, row in df.iterrows():
        raw_text = str(row["Tekst"])
        cleaned = clean_text(raw_text)
        chunks = chunk_text(cleaned, chunk_size=300)
        for c in chunks:
            all_chunks.append({"Auteur": row["Auteur"], "Chunk": c})
    
    chunks_df = pd.DataFrame(all_chunks)
    chunks_df.to_csv(CHUNKED_TEXTS_FILE, index=False)
    print(f"✅ Auteur chunks opgeslagen in {CHUNKED_TEXTS_FILE}")
    return chunks_df

# --- Test ---
if __name__ == "__main__":
    print("\n🔍 Testquery: 'magisch avontuur'")
    results = zoek_boeken("magisch avontuur")
    for r in results:
        print(f"{r['Titel']} - {r['Auteur']} (score: {r['Score']})")

    # Auteursteksten preprocessen
    preprocess_author_texts()
