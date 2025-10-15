import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BOOKS_FILE = "books.csv"

# --- Data inladen ---
def load_books():
    if os.path.exists(BOOKS_FILE):
        df = pd.read_csv(BOOKS_FILE)
        df = df.fillna("")
        return df
    else:
        return pd.DataFrame(columns=["Titel", "Auteur", "Genre", "Beschrijving"])

# --- Aanbevelingsfunctie ---
def maak_aanbevelingen(titel, top_n=5):
    books_df = load_books()

    if books_df.empty:
        print("⚠️ Geen boeken gevonden in books.csv")
        return []

    # Combineer genre en beschrijving voor rijkere context
    books_df["combined"] = books_df["Genre"].astype(str) + " " + books_df["Beschrijving"].astype(str)

    # TF-IDF vectorisatie
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(books_df["combined"])

    # Cosine similarity berekenen
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    if titel not in books_df["Titel"].values:
        print(f"⚠️ Boek '{titel}' niet gevonden.")
        return []

    idx = books_df.index[books_df["Titel"] == titel][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    results = []
    for i, score in sim_scores:
        results.append({
            "Titel": books_df.iloc[i]["Titel"],
            "Auteur": books_df.iloc[i]["Auteur"],
            "Genre": books_df.iloc[i]["Genre"],
            "Score": round(score, 3)
        })

    return results

# --- Test ---
if __name__ == "__main__":
    boek = input("Voer een boektitel in voor aanbevelingen: ")
    aanbevelingen = maak_aanbevelingen(boek)
    if aanbevelingen:
        print("\n📚 Aanbevolen boeken:")
        for r in aanbevelingen:
            print(f"- {r['Titel']} ({r['Auteur']}) | Genre: {r['Genre']} | Score: {r['Score']}")
