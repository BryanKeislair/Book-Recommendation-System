# generate_embeddings.py
import os
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np

BOOKS_FILE = "books.csv"
EMBEDDINGS_FILE = "book_embeddings.npy"
META_FILE = "book_meta.pkl"  # bewaart order/titels etc.
MODEL_NAME = "all-MiniLM-L6-v2"  # compact & snel, goede balans

def load_books():
    if os.path.exists(BOOKS_FILE):
        df = pd.read_csv(BOOKS_FILE).fillna("")
        expected_columns = ["Titel", "Auteur", "Genre", "Beschrijving", "Cover"]
        for col in expected_columns:
            if col not in df.columns:
                df[col] = ""
        return df[expected_columns]
    else:
        raise FileNotFoundError(f"{BOOKS_FILE} niet gevonden. Voeg eerst boeken toe.")

def build_corpus(df):
    # Combineer velden tot rijke representatie
    # (je kunt hier makkelijk extra velden toevoegen in toekomst)
    corpus = (df["Titel"].astype(str) + " . " +
              df["Auteur"].astype(str) + " . " +
              df["Genre"].astype(str) + " . " +
              df["Beschrijving"].astype(str))
    return corpus.tolist()

def main():
    print("🔁 Laden boeken...")
    df = load_books()
    if df.empty:
        print("⚠️ Geen boeken in books.csv - niets te doen.")
        return

    corpus = build_corpus(df)
    print(f"🧠 Laden model {MODEL_NAME} (éénmalig, kan even duren)...")
    model = SentenceTransformer(MODEL_NAME)

    print("✨ Embeddings genereren...")
    embeddings = model.encode(corpus, show_progress_bar=True, convert_to_numpy=True)

    # Opslaan (numpy + meta)
    np_path = EMBEDDINGS_FILE
    meta_path = META_FILE
    np.save(np_path, embeddings)
    joblib.dump({
        "titles": df["Titel"].tolist(),
        "index": df.index.tolist()
    }, meta_path)

    print(f"✅ Embeddings opgeslagen: {np_path}")
    print(f"✅ Meta opgeslagen: {meta_path}")
    print("Klaar.")

if __name__ == "__main__":
    main()
