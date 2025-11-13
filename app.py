# app.py
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
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

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

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    corpus = (
        df["Titel"].astype(str) + " . " +
        df["Auteur"].astype(str) + " . " +
        df["Genre"].astype(str) + " . " +
        df["Beschrijving"].astype(str)
    ).tolist()
    embeddings = model.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
    np.save(EMBEDDINGS_FILE, embeddings)
    joblib.dump({"titles": df["Titel"].tolist()}, META_FILE)

def persoonlijke_aanbevelingen(user_df, books_df, top_n=5, include_google=True):
    if user_df is None or user_df.empty:
        return []

    fav_books = user_df[user_df["Status"] == "Favoriet"]["Titel"].tolist()
    if not fav_books:
        return []

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    local_results = []
    google_results = []

    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(META_FILE):
        embeddings = np.load(EMBEDDINGS_FILE)
        meta = joblib.load(META_FILE)
        titles_meta = meta.get("titles", [])
        title_to_idx = {t: i for i, t in enumerate(titles_meta)}

        fav_idx = [title_to_idx[t] for t in fav_books if t in title_to_idx]
        if fav_idx:
            mean_vector = np.mean(embeddings[fav_idx], axis=0, keepdims=True)
            sims = cosine_similarity(mean_vector, embeddings).flatten()
            ranked_idx = sims.argsort()[::-1]
            for i in ranked_idx:
                book_title = books_df.iloc[i]["Titel"]
                # sla favorieten zelf over
                if book_title in fav_books:
                    continue
                local_results.append({
                    "Titel": book_title,
                    "Auteur": books_df.iloc[i]["Auteur"],
                    "Genre": books_df.iloc[i]["Genre"],
                    "Beschrijving": books_df.iloc[i].get("Beschrijving", ""),
                    "Cover": books_df.iloc[i].get("Cover", ""),
                    "Bron": "📘 Eigen bibliotheek",
                    "Score": round(float(sims[i] * 100), 2)
                })
                if len(local_results) >= top_n:
                    break
    else:
        mean_vector = None

    if include_google and fav_books:
        # voorkom duplicates
        seen_titles = set([r["Titel"].lower() for r in local_results] + [t.lower() for t in fav_books])
        if mean_vector is None:
            fav_texts = []
            for f in fav_books:
                match = books_df[books_df["Titel"] == f]
                if not match.empty:
                    descr = match.iloc[0].get("Beschrijving", "")
                    auth = match.iloc[0].get("Auteur", "")
                    gen = match.iloc[0].get("Genre", "")
                    fav_texts.append(f"{f}. {auth}. {gen}. {descr}")
                else:
                    fav_texts.append(f)
            if fav_texts:
                fav_vecs = model.encode(fav_texts, convert_to_numpy=True)
                mean_vector = np.mean(fav_vecs, axis=0, keepdims=True)

        if mean_vector is not None:
            for fav in fav_books[:3]:
                try:
                    q = requests.utils.quote(fav)
                    url = f"https://www.googleapis.com/books/v1/volumes?q={q}&maxResults=6"
                    resp = requests.get(url, timeout=8)
                    data = resp.json()
                    items = data.get("items", []) if isinstance(data, dict) else []
                    for item in items:
                        info = item.get("volumeInfo", {})
                        title = info.get("title", "Onbekende titel")
                        if title.lower() in seen_titles:
                            continue
                        authors = ", ".join(info.get("authors", ["Onbekende auteur"]))
                        genre = ", ".join(info.get("categories", ["Onbekend genre"]))
                        description = info.get("description", "Geen beschrijving beschikbaar")
                        cover = info.get("imageLinks", {}).get("thumbnail", "")

                        text = f"{title}. {authors}. {genre}. {description}"
                        gb_vec = model.encode(text, convert_to_numpy=True)
                        score = cosine_similarity(mean_vector, gb_vec.reshape(1, -1))[0][0]
                        score_pct = round(float(score * 100), 2)

                        google_results.append({
                            "Titel": title,
                            "Auteur": authors,
                            "Genre": genre,
                            "Beschrijving": description,
                            "Cover": cover,
                            "Bron": "🌐 Google Books",
                            "Score": score_pct
                        })
                        seen_titles.add(title.lower())
                except Exception as e:
                    # log en ga verder
                    print(f"[persoonlijke_aanbevelingen] Fout bij Google Books ophalen voor '{fav}': {e}")
                    continue

    # Combineer en sorteer
    combined = sorted(local_results + google_results, key=lambda x: x["Score"], reverse=True)
    return combined[: max(top_n + 5, top_n)]

def score_to_color(score_pct: float) -> str:
    """
    Return hex color based on fixed ranges:
      <40 -> red (#e53935)
      40-70 -> yellow (#f4b400)
      >70 -> green (#43a047)
    """
    if score_pct < 40:
        return "#e53935"  # red
    if score_pct <= 70:
        return "#f4b400"  
    return "#43a047"      # green

def render_score_bar(score_pct: float):
    """Return HTML snippet for small horizontal bar with color based on score."""
    color = score_to_color(score_pct)
    safe_pct = max(0, min(100, float(score_pct)))
    bar_html = (
        f"<div style='background:#e6e6e6;border-radius:8px;width:100%;height:14px;'>"
        f"<div style='background:{color};width:{safe_pct}%;height:14px;border-radius:8px;'></div>"
        f"</div>"
        f"<div style='font-size:12px;margin-top:4px;'>💡 Overeenkomst: <strong>{safe_pct}%</strong></div>"
    )
    return bar_html

# --- Streamlit App Start ---
st.set_page_config(page_title="Leesplatform", page_icon="📚", layout="wide")
st.title("📚 Leesplatform")

if "books_df" not in st.session_state:
    st.session_state.books_df = load_books()
if "user_df" not in st.session_state:
    st.session_state.user_df = load_user_data()
if "gb_items" not in st.session_state:
    st.session_state["gb_items"] = []

# --- Handmatig boek toevoegen ---
with st.expander("📘 Handmatig boek toevoegen (optioneel)"):
    with st.form("add_book_form"):
        form_title = st.text_input("Titel*")
        form_author = st.text_input("Auteur*")
        form_genre = st.text_input("Genre")
        form_description = st.text_area("Beschrijving")
        form_cover_file = st.file_uploader("Coverafbeelding (jpg/png)", type=["jpg", "jpeg", "png"])
        submitted = st.form_submit_button("Toevoegen")
        if submitted:
            if form_title.strip() and form_author.strip():
                os.makedirs(COVERS_DIR, exist_ok=True)
                cover_path = ""
                if form_cover_file is not None:
                    cover_path = os.path.join(COVERS_DIR, f"{form_title.strip().replace(' ', '_')}.png")
                    with open(cover_path, "wb") as f:
                        f.write(form_cover_file.getbuffer())

                new_row = pd.DataFrame([{
                    "Titel": form_title.strip(),
                    "Auteur": form_author.strip(),
                    "Genre": form_genre.strip(),
                    "Beschrijving": form_description.strip(),
                    "Cover": cover_path
                }])

                st.session_state.books_df = pd.concat([st.session_state.books_df, new_row], ignore_index=True)
                save_books(st.session_state.books_df)
                update_embeddings_ui()
                st.success(f"✅ '{form_title}' toegevoegd aan je bibliotheek!")
            else:
                st.error("Titel en auteur zijn verplicht.")

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

if st.session_state.get("gb_items"):
    st.markdown("**Zoekresultaten (Google Books)**")
    for i, info in enumerate(st.session_state["gb_items"]):
        with st.container():
            cols = st.columns([1, 3])
            if info["cover"]:
                cols[0].image(info["cover"], width=100)
            else:
                cols[0].write("📕 Geen cover")
            cols[1].markdown(f"**{info['title']}**  \n*Auteur:* {info['authors']}  \n*Genre:* {info['genre']}")
            cols[1].caption(info["description"][:300] + ("..." if len(info["description"]) > 300 else ""))
            if cols[1].button(f"➕ Voeg toe: {info['title']}", key=f"add_gb_{i}"):
                exists = ((st.session_state.books_df["Titel"] == info["title"]) &
                          (st.session_state.books_df["Auteur"] == info["authors"])).any()
                if exists:
                    st.warning(f"'{info['title']}' van {info['authors']} staat al in je bibliotheek.")
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
                    st.success(f"✅ '{info['title']}' toegevoegd aan jouw bibliotheek!")
                    try:
                        st.session_state["gb_items"].pop(i)
                    except Exception:
                        st.session_state["gb_items"] = [it for j, it in enumerate(st.session_state.get("gb_items", [])) if j != i]
                    st.rerun()

st.subheader("📖 Boeken")
filter_optie = st.radio("📚 Toon boeken:", ["Alle boeken", "Favorieten", "Gelezen"], horizontal=True)
books_df = st.session_state.books_df.copy()
user_df = st.session_state.user_df.copy()

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
        cover_path = row.get("Cover", "")
        if isinstance(cover_path, str) and cover_path.strip() and (os.path.exists(cover_path) or cover_path.startswith("http")):
            cols[0].image(cover_path, width=120)
        else:
            cols[0].write("📕 Geen coverafbeelding")
        cols[1].markdown(
            f"**{row['Titel']}**  \n"
            f"*Auteur:* {row['Auteur']}  \n"
            f"*Genre:* {row['Genre']}  \n"
            f"{row['Beschrijving'] if row.get('Beschrijving') else ''}"
        )

        with cols[1]:
            col1, col2, col3 = st.columns([1, 1, 1])
            huidige_status = user_df[user_df["Titel"] == row["Titel"]]["Status"].values
            status = huidige_status[0] if (isinstance(huidige_status, np.ndarray) and huidige_status.size > 0) else None

            if status == "Gelezen":
                if col1.button("↩️ Ongelezen", key=f"unread_{i}"):
                    user_df = user_df[user_df["Titel"] != row["Titel"]]
                    save_user_data(user_df)
                    st.session_state.user_df = user_df
                    st.rerun()
            else:
                if col1.button("📘 Gelezen", key=f"read_{i}"):
                    user_df = user_df[user_df["Titel"] != row["Titel"]]
                    user_df.loc[len(user_df)] = [row["Titel"], "Gelezen"]
                    save_user_data(user_df)
                    st.session_state.user_df = user_df
                    st.rerun()

            if status == "Favoriet":
                if col2.button("💔 Favoriet verwijderen", key=f"unfav_{i}"):
                    user_df = user_df[user_df["Titel"] != row["Titel"]]
                    save_user_data(user_df)
                    st.session_state.user_df = user_df
                    st.rerun()
            else:
                if col2.button("💖 Favoriet", key=f"fav_{i}"):
                    user_df = user_df[user_df["Titel"] != row["Titel"]]
                    user_df.loc[len(user_df)] = [row["Titel"], "Favoriet"]
                    save_user_data(user_df)
                    st.session_state.user_df = user_df
                    st.rerun()

            # Verwijder met bevestiging
            with col3:
                if st.checkbox(f"Bevestig verwijdering van '{row['Titel']}'", key=f"confirm_{i}"):
                    if st.button("🗑️ Verwijder boek", key=f"delete_{i}"):
                        st.session_state.books_df = st.session_state.books_df[st.session_state.books_df["Titel"] != row["Titel"]]
                        save_books(st.session_state.books_df)
                        update_embeddings_ui()
                        st.success(f"📕 '{row['Titel']}' is verwijderd uit je bibliotheek.")
                        st.rerun()

        st.divider()

st.subheader("🎯 Persoonlijke aanbevelingen")
persoonlijke_recs = persoonlijke_aanbevelingen(user_df, st.session_state.books_df, top_n=5, include_google=True)
if not persoonlijke_recs:
    st.info("Markeer enkele boeken als favoriet om persoonlijke aanbevelingen te krijgen.")
else:
    for rec in persoonlijke_recs:
        cols = st.columns([1, 4])
        # cover
        if isinstance(rec.get("Cover", ""), str) and rec.get("Cover", "").strip():
            try:
                cols[0].image(rec["Cover"], width=100)
            except Exception:
                cols[0].write("📕 Geen cover")
        else:
            cols[0].write("📕 Geen cover")
        # text
        cols[1].markdown(
            f"**{rec['Titel']}**  \n"
            f"*Auteur:* {rec.get('Auteur', '')}  \n"
            f"*Genre:* {rec.get('Genre','')}  \n"
            f"*Bron:* {rec.get('Bron','📘 Eigen bibliotheek')}"
        )
        bar_html = render_score_bar(rec.get("Score", 0))
        cols[1].markdown(bar_html, unsafe_allow_html=True)

        if rec.get("Bron") == "🌐 Google Books":
            if st.button(f"➕ Voeg toe aan bibliotheek: {rec['Titel']}", key=f"addrec_{rec['Titel']}"):
                exists = ((st.session_state.books_df["Titel"] == rec["Titel"]) &
                          (st.session_state.books_df["Auteur"] == rec["Auteur"])).any()
                if exists:
                    st.warning(f"'{rec['Titel']}' staat al in je bibliotheek.")
                else:
                    new_row = pd.DataFrame([{
                        "Titel": rec["Titel"],
                        "Auteur": rec.get("Auteur", ""),
                        "Genre": rec.get("Genre", ""),
                        "Beschrijving": rec.get("Beschrijving", ""),
                        "Cover": rec.get("Cover", "")
                    }])
                    st.session_state.books_df = pd.concat([st.session_state.books_df, new_row], ignore_index=True)
                    save_books(st.session_state.books_df)
                    update_embeddings_ui()
                    st.success(f"✅ '{rec['Titel']}' toegevoegd aan je bibliotheek!")
                    st.rerun()
        st.divider()

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

if st.session_state.get("embedding_update_done", False):
    st.success("✅ Boekenlijst succesvol bijgewerkt!")
    st.session_state["embedding_update_done"] = False
