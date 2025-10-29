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

# --- Embeddings update ---
def update_embeddings_ui():
    st.info("🔄 Boekenlijst wordt bijgewerkt... even geduld aub ⏳")

    def _run():
        update_embeddings()
        st.session_state["embedding_update_done"] = True

    threading.Thread(target=_run, daemon=True).start()

def update_embeddings():
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
def persoonlijke_aanbevelingen(user_df, books_df, top_n=5, include_google=True):
    """Combineert lokale aanbevelingen met inhoudelijk berekende Google Books-scores (in procenten)."""
    if user_df.empty:
        return []

    fav_books = user_df[user_df["Status"] == "Favoriet"]["Titel"].tolist()
    if not fav_books:
        return []

    local_results = []
    google_results = []
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # --- Lokale aanbevelingen ---
    if os.path.exists(EMBEDDINGS_FILE):
        embeddings = np.load(EMBEDDINGS_FILE)
        meta = joblib.load(META_FILE)
        titles = meta["titles"]
        title_to_idx = {t: i for i, t in enumerate(titles)}
        fav_idx = [title_to_idx[t] for t in fav_books if t in title_to_idx]
        if fav_idx:
            mean_vector = np.mean(embeddings[fav_idx], axis=0, keepdims=True)
            sims = cosine_similarity(mean_vector, embeddings).flatten()
            top_idx = sims.argsort()[::-1]
            for i in top_idx:
                book_title = books_df.iloc[i]["Titel"]
                if book_title not in fav_books:
                    local_results.append({
                        "Titel": book_title,
                        "Auteur": books_df.iloc[i]["Auteur"],
                        "Genre": books_df.iloc[i]["Genre"],
                        "Cover": books_df.iloc[i]["Cover"],
                        "Bron": "📘 Eigen bibliotheek",
                        "Score": round(float(sims[i] * 100), 2)
                    })
                if len(local_results) >= top_n:
                    break

    # --- Google Books aanbevelingen ---
    if include_google and fav_books:
        seen_titles = set([b["Titel"].lower() for b in local_results] + [t.lower() for t in fav_books])
        if 'mean_vector' in locals():
            for fav in fav_books[:3]:
                try:
                    query = requests.utils.quote(fav)
                    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=5"
                    response = requests.get(url, timeout=10)
                    data = response.json()
                    for item in data.get("items", []):
                        info = item.get("volumeInfo", {})
                        title = info.get("title", "Onbekende titel")
                        if title.lower() in seen_titles:
                            continue
                        authors = ", ".join(info.get("authors", ["Onbekende auteur"]))
                        genre = ", ".join(info.get("categories", ["Onbekend genre"]))
                        description = info.get("description", "Geen beschrijving beschikbaar")
                        cover = info.get("imageLinks", {}).get("thumbnail", "")
                        text = f"{title}. {authors}. {genre}. {description}"
                        google_vec = model.encode(text, convert_to_numpy=True)
                        score = cosine_similarity(mean_vector, google_vec.reshape(1, -1))[0][0]
                        score_percent = round(float(score * 100), 2)
                        google_results.append({
                            "Titel": title,
                            "Auteur": authors,
                            "Genre": genre,
                            "Beschrijving": description,
                            "Cover": cover,
                            "Bron": "🌐 Google Books",
                            "Score": score_percent
                        })
                        seen_titles.add(title.lower())
                except Exception as e:
                    print(f"Fout bij ophalen Google Books aanbeveling: {e}")
                    continue

    combined = local_results + google_results
    combined = sorted(combined, key=lambda x: x["Score"], reverse=True)
    return combined[:top_n + 5]

# --- Streamlit setup ---
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
                st.session_state.books_df = pd.concat([st.session_state.books_df, new_row], ignore_index=True)
                save_books(st.session_state.books_df)
                update_embeddings_ui()
                st.success(f"✅ '{title}' toegevoegd aan je bibliotheek!")
            else:
                st.error("Titel en auteur zijn verplicht.")

# --- Google Books zoeken ---
st.subheader("🌐 Boeken toevoegen vanuit Google Books")
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
            st.error(f"Fout bij ophalen: {e}")
            data = {}
        items = data.get("items", [])
        gb_items = []
        for item in items:
            info = item.get("volumeInfo", {})
            gb_items.append({
                "title": info.get("title", "Onbekende titel"),
                "authors": ", ".join(info.get("authors", ["Onbekende auteur"])),
                "genre": ", ".join(info.get("categories", ["Onbekend genre"])),
                "description": info.get("description", "Geen beschrijving beschikbaar"),
                "cover": info.get("imageLinks", {}).get("thumbnail", "")
            })
        st.session_state["gb_items"] = gb_items

if st.session_state.get("gb_items"):
    for i, info in enumerate(st.session_state["gb_items"]):
        with st.container():
            cols = st.columns([1, 3])
            if info["cover"]:
                cols[0].image(info["cover"], width=100)
            cols[1].markdown(f"**{info['title']}**  \n*Auteur:* {info['authors']}  \n*Genre:* {info['genre']}")
            cols[1].caption(info["description"][:300] + ("..." if len(info["description"]) > 300 else ""))
            if st.button(f"➕ Voeg toe: {info['title']}", key=f"add_gb_{i}"):
                exists = ((st.session_state.books_df["Titel"] == info["title"]) &
                          (st.session_state.books_df["Auteur"] == info["authors"])).any()
                if exists:
                    st.warning(f"'{info['title']}' bestaat al in je bibliotheek.")
                else:
                    new_row = pd.DataFrame([{
                        "Titel": info["title"],
                        "Auteur": info["authors"],
                        "Genre": info["genre"],
                        "Beschrijving": info["description"],
                        "Cover": info["cover"]
                    }])
                    st.session_state.books_df = pd.concat([st.session_state.books_df, new_row], ignore_index=True)
                    save_books(st.session_state.books_df)
                    update_embeddings_ui()
                    st.success(f"✅ '{info['title']}' toegevoegd!")
                    st.rerun()

# --- Boekenlijst ---
st.subheader("📖 Boeken")
filter_optie = st.radio("📚 Toon boeken:", ["Alle boeken", "Favorieten", "Gelezen"], horizontal=True)
books_df = st.session_state.books_df
user_df = st.session_state.user_df

if filter_optie == "Favorieten":
    fav_titles = user_df[user_df["Status"] == "Favoriet"]["Titel"].tolist()
    books_df = books_df[books_df["Titel"].isin(fav_titles)]
elif filter_optie == "Gelezen":
    read_titles = user_df[user_df["Status"] == "Gelezen"]["Titel"].tolist()
    books_df = books_df[books_df["Titel"].isin(read_titles)]

if books_df.empty:
    st.info("Nog geen boeken toegevoegd.")
else:
    for i, row in books_df.iterrows():
        cols = st.columns([1, 4])
        if isinstance(row["Cover"], str) and row["Cover"].strip() and (
            os.path.exists(row["Cover"]) or row["Cover"].startswith("http")
        ):
            cols[0].image(row["Cover"], width=120)
        else:
            cols[0].write("📕 Geen cover")
        cols[1].markdown(f"**{row['Titel']}**  \n*Auteur:* {row['Auteur']}  \n*Genre:* {row['Genre']}  \n{row['Beschrijving']}")
        with cols[1]:
            c1, c2, c3 = st.columns([1, 1, 1])
            status_vals = user_df[user_df["Titel"] == row["Titel"]]["Status"].values
            status = status_vals[0] if len(status_vals) > 0 else None

            # Toggle "Gelezen"
            if status == "Gelezen":
                if c1.button("↩️ Ongelezen", key=f"unread_{i}"):
                    user_df = user_df[user_df["Titel"] != row["Titel"]]
                    save_user_data(user_df)
                    st.session_state.user_df = user_df
                    st.rerun()
            else:
                if c1.button("📘 Gelezen", key=f"read_{i}"):
                    user_df = user_df[user_df["Titel"] != row["Titel"]]
                    user_df.loc[len(user_df)] = [row["Titel"], "Gelezen"]
                    save_user_data(user_df)
                    st.session_state.user_df = user_df
                    st.rerun()

            # Toggle "Favoriet"
            if status == "Favoriet":
                if c2.button("💔 Unfavoriet", key=f"unfav_{i}"):
                    user_df = user_df[user_df["Titel"] != row["Titel"]]
                    save_user_data(user_df)
                    st.session_state.user_df = user_df
                    st.rerun()
            else:
                if c2.button("💖 Favoriet", key=f"fav_{i}"):
                    user_df = user_df[user_df["Titel"] != row["Titel"]]
                    user_df.loc[len(user_df)] = [row["Titel"], "Favoriet"]
                    save_user_data(user_df)
                    st.session_state.user_df = user_df
                    st.rerun()

            # Verwijderen
            if c3.checkbox(f"Bevestig verwijdering van '{row['Titel']}'", key=f"conf_{i}"):
                if c3.button("🗑️ Verwijder", key=f"del_{i}"):
                    st.session_state.books_df = st.session_state.books_df[
                        st.session_state.books_df["Titel"] != row["Titel"]
                    ]
                    save_books(st.session_state.books_df)
                    update_embeddings_ui()
                    st.success(f"📕 '{row['Titel']}' verwijderd.")
                    st.rerun()
        st.divider()

# --- Persoonlijke aanbevelingen ---
st.subheader("🎯 Persoonlijke aanbevelingen")
persoonlijke_recs = persoonlijke_aanbevelingen(user_df, st.session_state.books_df)
if not persoonlijke_recs:
    st.info("Markeer enkele boeken als favoriet om aanbevelingen te krijgen.")
else:
    for rec in persoonlijke_recs:
        cols = st.columns([1, 4])
        if rec["Cover"]:
            cols[0].image(rec["Cover"], width=100)
        cols[1].markdown(
            f"**{rec['Titel']}**  \n"
            f"*Auteur:* {rec['Auteur']}  \n"
            f"*Genre:* {rec['Genre']}  \n"
            f"*Bron:* {rec['Bron']}  \n"
            f"💡 *Overeenkomst:* {rec['Score']}%"
        )
        if rec["Bron"] == "🌐 Google Books":
            if st.button(f"➕ Voeg toe aan bibliotheek: {rec['Titel']}", key=f"addrec_{rec['Titel']}"):
                exists = ((st.session_state.books_df["Titel"] == rec["Titel"]) &
                          (st.session_state.books_df["Auteur"] == rec["Auteur"])).any()
                if exists:
                    st.warning(f"'{rec['Titel']}' staat al in je bibliotheek.")
                else:
                    new_row = pd.DataFrame([{
                        "Titel": rec["Titel"],
                        "Auteur": rec["Auteur"],
                        "Genre": rec["Genre"],
                        "Beschrijving": rec.get("Beschrijving", ""),
                        "Cover": rec["Cover"]
                    }])
                    st.session_state.books_df = pd.concat(
                        [st.session_state.books_df, new_row],
                        ignore_index=True
                    )
                    save_books(st.session_state.books_df)
                    update_embeddings_ui()
                    st.success(f"✅ '{rec['Titel']}' toegevoegd!")
                    st.rerun()
        st.divider()

# --- Auteurherkenning ---
st.subheader("✍️ Auteurherkenning")
if os.path.exists("author_model.pkl") and os.path.exists("author_vectorizer.pkl"):
    model = joblib.load("author_model.pkl")
    vectorizer = joblib.load("author_vectorizer.pkl")
    text = st.text_area("Voer een tekstfragment in:")
    if st.button("🔍 Voorspel auteur"):
        if text.strip():
            X_vec = vectorizer.transform([text])
            probs = model.predict_proba(X_vec)[0]
            authors = model.classes_
            pred = authors[probs.argmax()]
            conf = round(probs.max() * 100, 2)
            st.success(f"🧠 Waarschijnlijk geschreven door **{pred}** ({conf}%)")
            df = pd.DataFrame({"Auteur": authors, "Zekerheid (%)": probs * 100}).sort_values("Zekerheid (%)", ascending=False)
            st.bar_chart(df.set_index("Auteur"))
        else:
            st.warning("Voer eerst een tekstfragment in.")
else:
    st.info("⚠️ Auteurmodel niet gevonden. Train eerst via `train_author_model.py`.")

# --- Embedding melding ---
if st.session_state.get("embedding_update_done", False):
    st.success("✅ Embeddings succesvol bijgewerkt!")
    st.session_state["embedding_update_done"] = False
