import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# --- Config ---
BOOKS_FILE = "books.csv"
VECTORS_FILE = "tfidf_vectors.npz"

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

# --- Preprocessing: TF-IDF ---
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)  # Engels standaard
tfidf_matrix = vectorizer.fit_transform(books_df["Beschrijving"].fillna(""))

# --- Train / Validatie / Test split ---
train_df, test_df = train_test_split(books_df, test_size=0.30, random_state=42)
valid_df, test_df = train_test_split(test_df, test_size=0.50, random_state=42)

print("📊 Dataset verdeling:")
print(f"Train: {len(train_df)} boeken")
print(f"Validatie: {len(valid_df)} boeken")
print(f"Test: {len(test_df)} boeken")

# --- Queryfunctie ---
def zoek_boeken(query, top_n=5):
    """Zoek boeken die het meest lijken op een query via cosine similarity."""
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

# --- Test ---
if __name__ == "__main__":
    print("\n🔍 Testquery: 'magisch avontuur'")
    for r in zoek_boeken("magisch avontuur"):
        print(f"{r['Titel']} - {r['Auteur']} (score: {r['Score']})")
