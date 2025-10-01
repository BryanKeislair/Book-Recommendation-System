import streamlit as st
import pandas as pd
import os

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
        return df[expected_columns]
    else:
        return pd.DataFrame(columns=["Titel", "Auteur", "Genre", "Beschrijving", "Cover"])

def save_books(df):
    df.to_csv(BOOKS_FILE, index=False)

# --- App start ---
st.set_page_config(page_title="AI Leesplatform", page_icon="📚", layout="wide")
st.title("📚 AI Leesplatform")

# --- Data inladen ---
if "books_df" not in st.session_state:
    st.session_state.books_df = load_books()

# --- Sectie: Boekenlijst ---
st.subheader("📖 Boeken")
if st.session_state.books_df.empty:
    st.info("Nog geen boeken toegevoegd.")
else:
    for i, row in st.session_state.books_df.iterrows():
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
