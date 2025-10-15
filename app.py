import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Config ---
BOOKS_FILE = "books.csv"
COVERS_DIR = "covers"

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

def maak_aanbevelingen(titel, df, top_n=5):
    """Zoek soortgelijke boeken o.b.v. beschrijving en genre."""
    if df.empty or titel not in df["Titel"].values:
        return []

    df["combined"] = df["Genre"].astype(str) + " " + df["Beschrijving"].astype(str)
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["combined"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    idx = df.index[df["Titel"] == titel][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    results = []
    for i, score in sim_scores:
        results.append({
            "Titel": df.iloc[i]["Titel"],
            "Auteur": df.iloc[i]["Auteur"],
            "Genre": df.iloc[i]["Genre"],
            "Cover": df.iloc[i]["Cover"],
            "Score": round(score, 3)
        })
    return results

# --- App start ---
st.set_page_config(page_title="AI Leesplatform", page_icon="📚", layout="wide")
st.title("📚 AI Leesplatform")

# --- Data inladen ---
if "books_df" not in st.session_state:
    st.session_state.books_df = load_books()

books_df = st.session_state.books_df

# --- Sectie: Boekenlijst ---
st.subheader("📖 Boeken")
if books_df.empty:
    st.info("Nog geen boeken toegevoegd.")
else:
    for i, row in books_df.iterrows():
        cols = st.columns([1, 4])
        cover_path = row["Cover"]

        if isinstance(cover_path, str) and cover_path.strip() and os.path.exists(cover_path):
            cols[0].image(cover_path, width=120)
        else:
            cols[0].write("📕")

        cols[1].markdown(
            f"**{row['Titel']}**  \n"
            f"*Auteur:* {row['Auteur']}  \n"
            f"*Genre:* {row['Genre']}  \n"
            f"{row['Beschrijving'] if row['Beschrijving'] else ''}"
        )

        if isinstance(cover_path, str) and cover_path.strip() and os.path.exists(cover_path):
            with st.expander("📸 Bekijk cover op volledig formaat"):
                st.image(cover_path, use_container_width=True)

        st.divider()

# --- Sectie: Boek toevoegen ---
with st.expander("➕ Nieuw boek toevoegen"):
    with st.form("add_book"):
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

                new_row = {
                    "Titel": title.strip(),
                    "Auteur": author.strip(),
                    "Genre": genre.strip(),
                    "Beschrijving": description.strip(),
                    "Cover": cover_path
                }

                st.session_state.books_df = pd.concat(
                    [st.session_state.books_df, pd.DataFrame([new_row])],
                    ignore_index=True
                )

                save_books(st.session_state.books_df)
                st.success(f"✅ '{title}' toegevoegd!")
            else:
                st.error("Titel en auteur zijn verplicht.")

# --- Sectie: Aanbevolen boeken ---
st.subheader("✨ Aanbevolen boeken")
if books_df.empty:
    st.info("Voeg eerst boeken toe om aanbevelingen te genereren.")
else:
    selected_title = st.selectbox("Kies een boek waarvoor je aanbevelingen wilt:", books_df["Titel"].tolist())

    if selected_title:
        aanbevelingen = maak_aanbevelingen(selected_title, books_df)

        if not aanbevelingen:
            st.warning("Geen vergelijkbare boeken gevonden.")
        else:
            st.write(f"📚 Boeken die lijken op **{selected_title}**:")

            for rec in aanbevelingen:
                cols = st.columns([1, 4])
                if isinstance(rec["Cover"], str) and rec["Cover"].strip() and os.path.exists(rec["Cover"]):
                    cols[0].image(rec["Cover"], width=100)
                else:
                    cols[0].write("📕")

                cols[1].markdown(
                    f"**{rec['Titel']}**  \n"
                    f"*Auteur:* {rec['Auteur']}  \n"
                    f"*Genre:* {rec['Genre']}  \n"
                    f"💡 *Overeenkomstscore:* {rec['Score']}"
                )
                st.divider()
