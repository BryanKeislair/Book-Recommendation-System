import streamlit as st
import pandas as pd
import os
import numpy as np
import joblib
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import threading

# --- Config ---
BOOKS_FILE = "books.csv"
USER_DATA_FILE = "user_data.csv"
COVERS_DIR = "covers"
EMBEDDINGS_FILE = "book_embeddings.npy"
META_FILE = "book_meta.pkl"

# --- Functies ---
def load_books():
    if os.path.exists(BOOKS_FILE):
        df = pd.read_csv(BOOKS_FILE)
        expected_columns = ["Titel", "Auteur", "Genre", "Beschrijving", "Cover"]
        for col in expected_columns:
            if col not in df.columns:
                df[col] = ""
        return df[expected_columns].fillna("")
    else:
        return pd.DataFrame(columns=["Titel", "Auteur", "Genre", "Beschrijving", "Cover"])

def save_books(df):
    df.to_csv(BOOKS_FILE, index=False)

def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        return pd.read_csv(USER_DATA_FILE)
    else:
        return pd.DataFrame(columns=["Titel", "Status"])

def save_user_data(df):
    df.to_csv(USER_DATA_FILE, index=False)

# --- Embeddings functie ---
def update_embeddings_ui():
    """Toont melding en start embedding-update in een aparte thread."""
    st.info("🔄 Boekenlijst wordt bijgewerkt... even geduld aub ⏳")

    def _run():
        update_embeddings()
        st.session_state["embedding_update_done"] = True

    threading.Thread(target=_run, daemon=True).start()

def update_embeddings():
    """Herberekent ALLE embeddings en slaat ze op."""
    df = load_books()
    if df.empty:
        return

    model = SentenceTransformer("all-MiniLM-L6-v2")
    corpus = (
        df["Titel"].astype(str) + " . " +
        df["Auteur"].astype(str) + " . " +
        df["Genre"].astype(str) + " . " +
        df["Beschrijving"].astype(str)
    )

    embeddings = model.encode(corpus, convert_to_numpy=True)
    np.save(EMBEDDINGS_FILE, embeddings)
    joblib.dump({"titles": df["Titel"].tolist()}, META_FILE)

# --- Aanbevelingsfuncties ---
def _maak_aanbevelingen_tfidf(titel, df, top_n=5):
    df_local = df.copy()
    df_local["combined"] = df_local["Genre"].astype(str) + " " + df_local["Beschrijving"].astype(str)
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df_local["combined"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    idx = df_local.index[df_local["Titel"] == titel][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

    results = []
    for i, score in sim_scores:
        results.append({
            "Titel": df_local.iloc[i]["Titel"],
            "Auteur": df_local.iloc[i]["Auteur"],
            "Genre": df_local.iloc[i]["Genre"],
            "Cover": df_local.iloc[i]["Cover"],
            "Score": round(score, 3)
        })
    return results

def maak_aanbevelingen_semantic(titel, df, top_n=5):
    if df.empty or titel not in df["Titel"].values:
        return []

    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(META_FILE):
        embeddings = np.load(EMBEDDINGS_FILE)
        meta = joblib.load(META_FILE)
        titles = meta["titles"]

        title_to_idx = {t: i for i, t in enumerate(titles)}
        if titel not in title_to_idx:
            return _maak_aanbevelingen_tfidf(titel, df, top_n)

        idx = title_to_idx[titel]
        sims = cosine_similarity(embeddings[idx:idx + 1], embeddings).flatten()
        top_idx = sims.argsort()[::-1][1: top_n + 1]

        results = []
        for i in top_idx:
            row = df.iloc[i]
            results.append({
                "Titel": row["Titel"],
                "Auteur": row["Auteur"],
                "Genre": row["Genre"],
                "Cover": row.get("Cover", ""),
                "Score": float(round(sims[i], 3))
            })
        return results
    else:
        return _maak_aanbevelingen_tfidf(titel, df, top_n)

def persoonlijke_aanbevelingen(user_df, books_df, top_n=5):
    if user_df.empty or not os.path.exists(EMBEDDINGS_FILE):
        return []

    fav_books = user_df[user_df["Status"] == "Favoriet"]["Titel"].tolist()
    if not fav_books:
        return []

    embeddings = np.load(EMBEDDINGS_FILE)
    meta = joblib.load(META_FILE)
    titles = meta["titles"]

    title_to_idx = {t: i for i, t in enumerate(titles)}
    fav_idx = [title_to_idx[t] for t in fav_books if t in title_to_idx]
    if not fav_idx:
        return []

    mean_vector = np.mean(embeddings[fav_idx], axis=0, keepdims=True)
    sims = cosine_similarity(mean_vector, embeddings).flatten()
    top_idx = sims.argsort()[::-1]

    results = []
    for i in top_idx:
        book_title = books_df.iloc[i]["Titel"]
        if book_title not in fav_books:
            results.append({
                "Titel": book_title,
                "Auteur": books_df.iloc[i]["Auteur"],
                "Genre": books_df.iloc[i]["Genre"],
                "Cover": books_df.iloc[i]["Cover"],
                "Score": float(round(sims[i], 3))
            })
        if len(results) >= top_n:
            break
    return results

# --- App start ---
st.set_page_config(page_title="Leesplatform", page_icon="📚", layout="wide")
st.title("📚 Leesplatform")

if "books_df" not in st.session_state:
    st.session_state.books_df = load_books()
if "user_df" not in st.session_state:
    st.session_state.user_df = load_user_data()

# --- Handmatig boek toevoegen ---
with st.expander("📘 Handmatig boek toevoegen (optioneel)"):
    with st.form("add_book_form"):
        title = st.text_input("Titel*")
        author = st.text_input("Auteur*")
        genre = st.text_input("Genre")
        description = st.text_area("Beschrijving")
        cover_file = st.file_uploader("Coverafbeelding (jpg/png)", type=["jpg", "jpeg", "png"])
        submitted = st.form_submit_button("Toevoegen")

        if submitted:
            if title.strip() and author.strip():
                os.makedirs(COVERS_DIR, exist_ok=True)
                cover_path = ""
                if cover_file is not None:
                    cover_path = os.path.join(COVERS_DIR, f"{title.strip().replace(' ', '_')}.png")
                    with open(cover_path, "wb") as f:
                        f.write(cover_file.getbuffer())

                new_row = pd.DataFrame([{
                    "Titel": title.strip(),
                    "Auteur": author.strip(),
                    "Genre": genre.strip(),
                    "Beschrijving": description.strip(),
                    "Cover": cover_path
                }])

                st.session_state.books_df = pd.concat(
                    [st.session_state.books_df, new_row],
                    ignore_index=True
                )
                save_books(st.session_state.books_df)
                update_embeddings_ui()
                st.success(f"✅ '{title}' toegevoegd aan je bibliotheek!")
            else:
                st.error("Titel en auteur zijn verplicht.")

# --- Google Books import (met persistente zoekresultaten) ---
st.subheader("🌐 Boeken importeren vanuit Google Books")
search_query = st.text_input("Zoek op titel, auteur of onderwerp:", key="gb_search")

if st.button("🔍 Zoek boeken", key="gb_search_btn"):
    if not search_query.strip():
        st.warning("Voer een zoekterm in.")
    else:
        url = f"https://www.googleapis.com/books/v1/volumes?q={search_query}&maxResults=12"
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
        except Exception as e:
            st.error(f"Fout bij ophalen van resultaten: {e}")
            data = {}

        items = data.get("items", [])
        gb_items = []
        for item in items:
            info = item.get("volumeInfo", {})
            title = info.get("title", "Onbekende titel")
            authors = ", ".join(info.get("authors", ["Onbekende auteur"]))
            genre = ", ".join(info.get("categories", ["Onbekend genre"]))
            description = info.get("description", "Geen beschrijving beschikbaar")
            cover = info.get("imageLinks", {}).get("thumbnail", "")

            gb_items.append({
                "title": title,
                "authors": authors,
                "genre": genre,
                "description": description,
                "cover": cover
            })

        st.session_state["gb_items"] = gb_items

# Resultaten tonen (blijven bestaan)
if st.session_state.get("gb_items"):
    items = st.session_state["gb_items"]
    for i, info in enumerate(items):
        title = info["title"]
        authors = info["authors"]
        genre = info["genre"]
        description = info["description"]
        cover = info["cover"]

        with st.container():
            cols = st.columns([1, 3])
            with cols[0]:
                if cover:
                    st.image(cover, width=100)
                else:
                    st.write("📕 Geen cover")
            with cols[1]:
                st.markdown(f"**{title}**  \n*Auteur:* {authors}  \n*Genre:* {genre}")
                st.caption(description[:300] + ("..." if len(description) > 300 else ""))

                if st.button(f"➕ Voeg toe: {title}", key=f"add_gb_{i}"):
                    exists = ((st.session_state.books_df["Titel"] == title) &
                              (st.session_state.books_df["Auteur"] == authors)).any()
                    if exists:
                        st.warning(f"'{title}' van {authors} staat al in je bibliotheek.")
                    else:
                        new_row = pd.DataFrame([{
                            "Titel": title,
                            "Auteur": authors,
                            "Genre": genre,
                            "Beschrijving": description,
                            "Cover": cover
                        }])
                        st.session_state.books_df = pd.concat(
                            [st.session_state.books_df, new_row],
                            ignore_index=True
                        )
                        save_books(st.session_state.books_df)
                        update_embeddings_ui()
                        st.success(f"✅ '{title}' toegevoegd aan jouw bibliotheek!")

                        # verwijder toegevoegd boek uit resultaten en herlaad
                        try:
                            st.session_state["gb_items"].pop(i)
                        except Exception:
                            pass
                        st.rerun()()

# --- Boekenlijst ---
st.subheader("📖 Boeken")
books_df = st.session_state.books_df
user_df = st.session_state.user_df

if books_df.empty:
    st.info("Nog geen boeken toegevoegd.")
else:
    for i, row in books_df.iterrows():
        cols = st.columns([1, 4])
        cover_path = row["Cover"]

        if isinstance(cover_path, str) and cover_path.strip() and (
            os.path.exists(cover_path) or cover_path.startswith("http")
        ):
            cols[0].image(cover_path, width=120)
        else:
            cols[0].write("📕 Geen coverafbeelding")

        cols[1].markdown(
            f"**{row['Titel']}**  \n"
            f"*Auteur:* {row['Auteur']}  \n"
            f"*Genre:* {row['Genre']}  \n"
            f"{row['Beschrijving'] if row['Beschrijving'] else ''}"
        )

        status = user_df[user_df["Titel"] == row["Titel"]]["Status"].values
        status_text = f"✅ ({status[0]})" if len(status) > 0 else ""

        with cols[1]:
            col1, col2 = st.columns(2)
            if col1.button(f"📘 Gelezen", key=f"read_{i}"):
                user_df = user_df[user_df["Titel"] != row["Titel"]]
                user_df.loc[len(user_df)] = [row["Titel"], "Gelezen"]
                save_user_data(user_df)
                st.session_state.user_df = user_df
                st.rerun()

            if col2.button(f"💖 Favoriet", key=f"fav_{i}"):
                user_df = user_df[user_df["Titel"] != row["Titel"]]
                user_df.loc[len(user_df)] = [row["Titel"], "Favoriet"]
                save_user_data(user_df)
                st.session_state.user_df = user_df
                st.rerun()

        st.caption(status_text)
        st.divider()

# --- Persoonlijke aanbevelingen ---
st.subheader("🎯 Persoonlijke aanbevelingen")
persoonlijke_recs = persoonlijke_aanbevelingen(user_df, books_df)
if not persoonlijke_recs:
    st.info("Markeer enkele boeken als favoriet om persoonlijke aanbevelingen te krijgen.")
else:
    for rec in persoonlijke_recs:
        cols = st.columns([1, 4])
        if isinstance(rec["Cover"], str) and rec["Cover"].strip() and (
            os.path.exists(rec["Cover"]) or rec["Cover"].startswith("http")
        ):
            cols[0].image(rec["Cover"], width=100)
        else:
            cols[0].write("📕 Geen cover")
        cols[1].markdown(
            f"**{rec['Titel']}**  \n"
            f"*Auteur:* {rec['Auteur']}  \n"
            f"*Genre:* {rec['Genre']}  \n"
            f"💡 *Score:* {rec['Score']}"
        )
        st.divider()

# --- Auteurherkenning ---
st.subheader("✍️ Auteurherkenning")
if os.path.exists("author_model.pkl") and os.path.exists("author_vectorizer.pkl"):
    model = joblib.load("author_model.pkl")
    vectorizer = joblib.load("author_vectorizer.pkl")

    input_text = st.text_area("Voer een tekstfragment in om de auteur te voorspellen:")

    if st.button("🔍 Voorspel auteur"):
        if input_text.strip():
            X_vec = vectorizer.transform([input_text])
            probs = model.predict_proba(X_vec)[0]
            authors = model.classes_
            best_idx = probs.argmax()
            predicted_author = authors[best_idx]
            confidence = probs[best_idx] * 100

            st.success(f"🧠 Waarschijnlijk geschreven door: **{predicted_author}** ({confidence:.2f}% zekerheid)")

            chart_data = pd.DataFrame({
                "Auteur": authors,
                "Zekerheid (%)": probs * 100
            }).sort_values("Zekerheid (%)", ascending=False)
            st.bar_chart(chart_data.set_index("Auteur"))
        else:
            st.warning("Voer eerst een tekstfragment in.")
else:
    st.info("⚠️ Auteurmodel niet gevonden. Train eerst het model via `train_author_model.py`.")

# --- Embedding melding ---
if st.session_state.get("embedding_update_done", False):
    st.success("✅ Boekenlijst succesvol bijgewerkt!")
    st.session_state["embedding_update_done"] = False
