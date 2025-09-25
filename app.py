import streamlit as st
import pandas as pd
import os

# --- Config ---
BOOKS_FILE = "books.csv"
COVERS_DIR = "covers"

# --- Functies ---
def load_books():
    if os.path.exists(BOOKS_FILE):
        return pd.read_csv(BOOKS_FILE)
    else:
        return pd.DataFrame(columns=["Title", "Author", "Genre", "Description", "Cover"])

def save_books(df):
    df.to_csv(BOOKS_FILE, index=False)

# --- App start ---
st.set_page_config(page_title="AI Leesplatform", page_icon="📚", layout="wide")
st.title("📚 AI Leesplatform")

# --- Data inladen ---
if "books_df" not in st.session_state:
    st.session_state.books_df = load_books()

books_df = st.session_state.books_df

# --- Sectie: Boekenlijst ---
st.header("📖 Boekenlijst")
if books_df.empty:
    st.info("Nog geen boeken toegevoegd.")
else:
    st.dataframe(books_df, use_container_width=True)

# --- Sectie: Boek toevoegen ---
st.header("➕ Nieuw boek toevoegen")
with st.form("add_book"):
    title = st.text_input("Titel*")
    author = st.text_input("Auteur*")
    genre = st.text_input("Genre")
    description = st.text_area("Beschrijving")
    cover_file = st.file_uploader("Coverafbeelding (jpg/png)", type=["jpg", "jpeg", "png"])
    submitted = st.form_submit_button("Toevoegen")

    if submitted:
        if title.strip() and author.strip():
            cover_path = ""
            if cover_file is not None:
                os.makedirs(COVERS_DIR, exist_ok=True)
                cover_path = f"{COVERS_DIR}/{title.strip().replace(' ', '_')}.png"
                with open(cover_path, "wb") as f:
                    f.write(cover_file.getbuffer())

            new_book = {"Title": title.strip(), 
                        "Author": author.strip(), 
                        "Genre": genre.strip(), 
                        "Description": description.strip(),
                        "Cover": cover_path
                    }
            # Voeg toe aan session_state dataframe
            st.session_state.books_df = pd.concat([books_df, pd.DataFrame([new_book])], ignore_index=True)
            save_books(st.session_state.books_df)
            st.success(f"✅ '{title}' toegevoegd!")
        else:
            st.error("Titel en auteur zijn verplicht.")
