# app.py
import streamlit as st
import pandas as pd
import os
import re
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

# --- File helpers ---
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

# --- Embeddings update (non-blocking UI) ---
def update_embeddings_ui():
    st.info("🔄 Embeddings worden bijgewerkt... even geduld aub ⏳")
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

# --- Series detection helpers (taalgevoelig) ---
SERIES_TOKENS = [
    r"\bdeel\b", r"\bbook\b", r"\bpart\b", r"\bvolume\b", r"\bvol\.\b", r"\bvol\b", r"\bserie\b",
    r"#\d+", r"\bchapter\b", r"\bepisode\b", r"\bbook\s*\d+\b"
]

ROMAN = r"\bM{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b"

def normalize_title_for_series(title: str) -> str:
    # Lowercase, remove punctuation except alphanumerics and spaces,
    # remove tokens like 'deel 1', 'book 2', roman numerals, trailing numbers
    t = title.lower()
    # remove parentheses content
    t = re.sub(r"\(.*?\)", "", t)
    # replace punctuation with space
    t = re.sub(r"[^\w\s]", " ", t)
    # remove extra spaces
    t = re.sub(r"\s+", " ", t).strip()
    # remove series indicators and trailing numbers for base
    t = re.sub(r"\b(deel|book|part|volume|vol|vol)\b\s*\d+", "", t)
    t = re.sub(r"\b(book|deel|part|volume|vol)\b", "", t)
    t = re.sub(r"\b" + ROMAN + r"\b", "", t)
    t = re.sub(r"\b\d+\b", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def is_series_match(title_a: str, title_b: str) -> bool:
    """
    Return True if titles likely belong to same series:
     - normalized base titles match significantly (one contains the other),
     - OR they share many words (author/title overlap),
     - OR one contains series token and base words are similar.
    This is deliberately permissive to catch typical series naming patterns.
    """
    if not title_a or not title_b:
        return False
    a_norm = normalize_title_for_series(title_a)
    b_norm = normalize_title_for_series(title_b)
    if not a_norm or not b_norm:
        return False
    # direct contains
    if a_norm in b_norm or b_norm in a_norm:
        return True
    # token presence + overlap
    a_has_token = any(re.search(tok, title_a, re.IGNORECASE) for tok in SERIES_TOKENS)
    b_has_token = any(re.search(tok, title_b, re.IGNORECASE) for tok in SERIES_TOKENS)
    if a_has_token or b_has_token:
        # compare word overlap
        a_words = set(a_norm.split())
        b_words = set(b_norm.split())
        if not a_words or not b_words:
            return False
        overlap = len(a_words & b_words) / max(1, min(len(a_words), len(b_words)))
        if overlap >= 0.5:
            return True
    # fallback: check if enough shared words (>60% of smaller title words)
    a_words = set(a_norm.split())
    b_words = set(b_norm.split())
    if a_words and b_words:
        overlap = len(a_words & b_words) / max(1, min(len(a_words), len(b_words)))
        if overlap >= 0.6:
            return True
    return False

# --- Scoring bonuses ---
def apply_bonuses(base_score_pct: float, candidate_author: str, fav_authors: list, candidate_title: str, fav_titles: list) -> float:
    """
    Apply:
      - authorsbonus: min(base * 0.10, 8)
      - seriesbonus: min(base * 0.15, 12)
      - combined bonus capped at 20
    Returns final_score (capped at 100).
    """
    score = float(base_score_pct)
    # author bonus if author matches any fav author exactly (case-insensitive)
    author_bonus = 0.0
    if candidate_author and fav_authors:
        if any(candidate_author.strip().lower() == fa.strip().lower() for fa in fav_authors):
            author_bonus = min(score * 0.10, 8.0)
    # series bonus if candidate title matches series pattern with any fav title
    series_bonus = 0.0
    for fav in fav_titles:
        if is_series_match(candidate_title, fav):
            series_bonus = min(score * 0.15, 12.0)
            break
    total_bonus = min(author_bonus + series_bonus, 20.0)
    final = min(score + total_bonus, 100.0)
    return round(final, 2)

# --- Recommendation function (local + google, with bonuses) ---
def persoonlijke_aanbevelingen(user_df, books_df, top_n=5, include_google=True):
    """
    Return combined sorted recommendations with Score in percent and bonuses applied internally.
    """
    if user_df is None or user_df.empty:
        return []

    fav_titles = user_df[user_df["Status"] == "Favoriet"]["Titel"].tolist()
    if not fav_titles:
        return []

    # collect fav authors for author bonus
    fav_authors = []
    for t in fav_titles:
        m = books_df[books_df["Titel"] == t]
        if not m.empty:
            fav_authors.append(m.iloc[0].get("Auteur", ""))
    # unique
    fav_authors = list(dict.fromkeys([a for a in fav_authors if a]))

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    local_results = []
    google_results = []

    mean_vector = None
    # local via precomputed embeddings
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(META_FILE):
        try:
            embeddings = np.load(EMBEDDINGS_FILE)
            meta = joblib.load(META_FILE)
            titles_meta = meta.get("titles", [])
            title_to_idx = {t: i for i, t in enumerate(titles_meta)}
            fav_idx = [title_to_idx[t] for t in fav_titles if t in title_to_idx]
            if fav_idx:
                mean_vector = np.mean(embeddings[fav_idx], axis=0, keepdims=True)
                sims = cosine_similarity(mean_vector, embeddings).flatten()
                ranked = sims.argsort()[::-1]
                for i in ranked:
                    candidate_title = books_df.iloc[i]["Titel"]
                    if candidate_title in fav_titles:
                        continue
                    candidate_author = books_df.iloc[i].get("Auteur", "")
                    base_pct = round(float(sims[i] * 100), 2)
                    final_pct = apply_bonuses(base_pct, candidate_author, fav_authors, candidate_title, fav_titles)
                    local_results.append({
                        "Titel": candidate_title,
                        "Auteur": candidate_author,
                        "Genre": books_df.iloc[i].get("Genre", ""),
                        "Beschrijving": books_df.iloc[i].get("Beschrijving", ""),
                        "Cover": books_df.iloc[i].get("Cover", ""),
                        "Bron": "📘 Eigen bibliotheek",
                        "Score": final_pct
                    })
                    if len(local_results) >= top_n:
                        break
        except Exception as e:
            print("[persoonlijke_aanbevelingen] fout bij lokale embeddings:", e)
            mean_vector = None

    # if no mean_vector from precomputed, build from favorite texts using model
    if mean_vector is None:
        fav_texts = []
        for f in fav_titles:
            match = books_df[books_df["Titel"] == f]
            if not match.empty:
                descr = match.iloc[0].get("Beschrijving", "")
                auth = match.iloc[0].get("Auteur", "")
                gen = match.iloc[0].get("Genre", "")
                fav_texts.append(f"{f}. {auth}. {gen}. {descr}")
            else:
                fav_texts.append(f)
        if fav_texts:
            try:
                fav_vecs = model.encode(fav_texts, convert_to_numpy=True)
                mean_vector = np.mean(fav_vecs, axis=0, keepdims=True)
            except Exception as e:
                print("[persoonlijke_aanbevelingen] fout bij mean_vector build:", e)
                mean_vector = None

    # Google Books recommendations (semantically scored) if mean_vector available
    if include_google and mean_vector is not None and fav_titles:
        seen_titles = set([r["Titel"].lower() for r in local_results] + [t.lower() for t in fav_titles])
        # use up-to-3 favorites as search terms (balance of relevance & speed)
        for fav in fav_titles[:3]:
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
                    # build text and compute embedding similarity
                    text = f"{title}. {authors}. {genre}. {description}"
                    try:
                        gb_vec = model.encode(text, convert_to_numpy=True)
                        base_score = cosine_similarity(mean_vector, gb_vec.reshape(1, -1))[0][0]
                        base_pct = round(float(base_score * 100), 2)
                    except Exception as e:
                        base_pct = 0.0
                        print("[persoonlijke_aanbevelingen] fout bij google vec:", e)
                    final_pct = apply_bonuses(base_pct, authors, fav_authors, title, fav_titles)
                    google_results.append({
                        "Titel": title,
                        "Auteur": authors,
                        "Genre": genre,
                        "Beschrijving": description,
                        "Cover": cover,
                        "Bron": "🌐 Google Books",
                        "Score": final_pct
                    })
                    seen_titles.add(title.lower())
            except Exception as e:
                print("[persoonlijke_aanbevelingen] fout Google Books ophalen:", e)
                continue

    combined = sorted(local_results + google_results, key=lambda x: x["Score"], reverse=True)
    # return a reasonable number
    return combined[: max(top_n + 5, top_n)]

# --- UI: color & bar helpers ---
def score_to_color(score_pct: float) -> str:
    """Return hex color for ranges: <40 red, 40-70 yellow, >70 green"""
    try:
        s = float(score_pct)
    except Exception:
        s = 0.0
    if s < 40:
        return "#e53935"  # red
    if s <= 70:
        return "#f4b400"  # yellow
    return "#43a047"      # green

def render_score_bar(score_pct: float):
    """Return HTML snippet for small horizontal bar with color based on score."""
    color = score_to_color(score_pct)
    safe_pct = max(0, min(100, float(score_pct)))
    bar_html = (
        f"<div style='background:#e6e6e6;border-radius:8px;width:100%;height:14px;'>"
        f"<div style='background:{color};width:{safe_pct}%;height:14px;border-radius:8px;'></div>"
        f"</div>"
        f"<div style='font-size:12px;margin-top:6px;'>💡 Overeenkomst: <strong>{safe_pct}%</strong></div>"
    )
    return bar_html

# --- Streamlit App start ---
st.set_page_config(page_title="Leesplatform", page_icon="📚", layout="wide")
st.title("📚 Leesplatform")

# ensure session state
if "books_df" not in st.session_state:
    st.session_state.books_df = load_books()
if "user_df" not in st.session_state:
    st.session_state.user_df = load_user_data()
if "gb_items" not in st.session_state:
    st.session_state["gb_items"] = []

# --- Handmatig toevoegen ---
with st.expander("📘 Handmatig boek toevoegen (optioneel)"):
    with st.form("add_book_form"):
        form_title = st.text_input("Titel*")
        form_author = st.text_input("Auteur*")
        form_genre = st.text_input("Genre")
        form_description = st.text_area("Beschrijving")
        form_cover = st.file_uploader("Coverafbeelding (jpg/png)", type=["jpg", "jpeg", "png"])
        submitted = st.form_submit_button("Toevoegen")
        if submitted:
            if form_title.strip() and form_author.strip():
                os.makedirs(COVERS_DIR, exist_ok=True)
                cover_path = ""
                if form_cover is not None:
                    cover_path = os.path.join(COVERS_DIR, f"{form_title.strip().replace(' ', '_')}.png")
                    with open(cover_path, "wb") as f:
                        f.write(form_cover.getbuffer())
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

# --- Google Books zoek (behoud) ---
st.subheader("🌐 Boeken importeren vanuit Google Books")
search_query = st.text_input("Zoek op titel, auteur of onderwerp:", key="gb_search")
if st.button("🔍 Zoek boeken", key="gb_search_btn"):
    if not search_query.strip():
        st.warning("Voer een zoekterm in.")
    else:
        url = f"https://www.googleapis.com/books/v1/volumes?q={search_query}&maxResults=12"
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()
        except Exception as e:
            st.error(f"Fout bij ophalen van resultaten: {e}")
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

# render persistent gb search results
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

# --- Book list: filter, toggles, delete ---
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
            f"*Genre:* {row.get('Genre','')}  \n"
            f"{row.get('Beschrijving','')}"
        )

        with cols[1]:
            col1, col2, col3 = st.columns([1, 1, 1])
            huidige_status = user_df[user_df["Titel"] == row["Titel"]]["Status"].values
            status = huidige_status[0] if (isinstance(huidige_status, np.ndarray) and huidige_status.size > 0) else None

            # Gelezen toggle
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

            # Favoriet toggle
            if status == "Favoriet":
                if col2.button("💔 Unfavoriet", key=f"unfav_{i}"):
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

# --- Persoonlijke aanbevelingen (met color bars) ---
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
        # meta text
        cols[1].markdown(
            f"**{rec['Titel']}**  \n"
            f"*Auteur:* {rec.get('Auteur','')}  \n"
            f"*Genre:* {rec.get('Genre','')}  \n"
            f"*Bron:* {rec.get('Bron','📘 Eigen bibliotheek')}"
        )
        # colored progress bar (HTML)
        bar_html = render_score_bar(rec.get("Score", 0.0))
        cols[1].markdown(bar_html, unsafe_allow_html=True)

        # add-to-library button for google results
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

# --- Auteurherkenning UI (ongewijzigd functionaliteit) ---
st.subheader("✍️ Auteurherkenning")
if os.path.exists("author_model.pkl") and os.path.exists("author_vectorizer.pkl"):
    model_author = joblib.load("author_model.pkl")
    vectorizer = joblib.load("author_vectorizer.pkl")

    input_text = st.text_area("Voer een tekstfragment in om de auteur te voorspellen:")
    if st.button("🔍 Voorspel auteur"):
        if input_text.strip():
            X_vec = vectorizer.transform([input_text])
            probs = model_author.predict_proba(X_vec)[0]
            authors = model_author.classes_
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

# --- Embedding update voltooid melding ---
if st.session_state.get("embedding_update_done", False):
    st.success("✅ Embeddings succesvol bijgewerkt!")
    st.session_state["embedding_update_done"] = False
