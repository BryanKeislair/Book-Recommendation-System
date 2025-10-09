# preprocess.py
import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk

# Zorg dat NLTK data aanwezig is (stopwoorden, tokenizers, enz.)
nltk.download("punkt")
nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Config ---
BOOKS_FILE = "books.csv"
AUTHORS_DIR = "auteurs"   # map waar je losse txt-bestanden per auteur zet
OUTPUT_DIR = "processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# 1. Boekbeschrijvingen vectoriseren
# -------------------------------
def preprocess_books():
    if not os.path.exists(BOOKS_FILE):
        print("⚠️ Geen books.csv gevonden.")
        return None

    df = pd.read_csv(BOOKS_FILE)

    if "Beschrijving" not in df.columns:
        print("⚠️ Geen beschrijvingen in dataset.")
        return None

    # Beschrijvingen vectoriseren met TF-IDF
    vectorizer = TfidfVectorizer(stop_words="dutch", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df["Beschrijving"].fillna(""))

    # Opslaan
    np.save(os.path.join(OUTPUT_DIR, "book_tfidf.npy"), tfidf_matrix.toarray())
    pd.DataFrame(vectorizer.get_feature_names_out()).to_csv(
        os.path.join(OUTPUT_DIR, "tfidf_features.csv"), index=False
    )
    print(f"✅ Boekenbeschrijvingen vectorized en opgeslagen in {OUTPUT_DIR}/")

    return tfidf_matrix, vectorizer


# -------------------------------
# 2. Auteursdata voorbereiden
# -------------------------------
def clean_text(text):
    # Lowercase, verwijder niet-letters
    text = text.lower()
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", "", text)
    tokens = word_tokenize(text, language="dutch")
    tokens = [t for t in tokens if t not in stopwords.words("dutch")]
    return " ".join(tokens)


def preprocess_authors():
    if not os.path.exists(AUTHORS_DIR):
        print("⚠️ Geen map 'auteurs' gevonden.")
        return None

    data = []
    for author in os.listdir(AUTHORS_DIR):
        author_dir = os.path.join(AUTHORS_DIR, author)
        if os.path.isdir(author_dir):
            for file in os.listdir(author_dir):
                if file.endswith(".txt"):
                    path = os.path.join(author_dir, file)
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()

                    # Tekst opdelen in stukken van ±300 woorden
                    words = text.split()
                    chunk_size = 300
                    for i in range(0, len(words), chunk_size):
                        chunk = " ".join(words[i:i+chunk_size])
                        cleaned = clean_text(chunk)
                        if cleaned.strip():
                            data.append({"Auteur": author, "Tekst": cleaned})

    df = pd.DataFrame(data)

    # Train/val/test split
    train, temp = train_test_split(df, test_size=0.3, stratify=df["Auteur"], random_state=42)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp["Auteur"], random_state=42)

    train.to_csv(os.path.join(OUTPUT_DIR, "authors_train.csv"), index=False)
    val.to_csv(os.path.join(OUTPUT_DIR, "authors_val.csv"), index=False)
    test.to_csv(os.path.join(OUTPUT_DIR, "authors_test.csv"), index=False)

    print(f"✅ Auteursdata opgesplitst in train/val/test in {OUTPUT_DIR}/")

    return train, val, test


# -------------------------------
# Main script
# -------------------------------
if __name__ == "__main__":
    print("🚀 Start preprocessing...")
    preprocess_books()
    preprocess_authors()
    print("🎉 Preprocessing afgerond!")
