import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# === Config ===
DATA_DIR = "data/authors"
MODEL_FILE = "author_model.pkl"
VECTORIZER_FILE = "author_vectorizer.pkl"

# === 1. Dataset inladen ===
data = []

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"❌ Map niet gevonden: {DATA_DIR}. Controleer of 'data/authors' bestaat.")

for file in os.listdir(DATA_DIR):
    if file.endswith(".txt"):
        author = file.replace(".txt", "")
        file_path = os.path.join(DATA_DIR, file)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

            # Splits tekst op basis van lege regels of regeleinden
            chunks = [chunk.strip() for chunk in text.splitlines() if chunk.strip()]

            for c in chunks:
                # Alleen fragmenten met voldoende woorden gebruiken
                if len(c.split()) > 5:  # iets lagere drempel voor kleinere teksten
                    data.append({"auteur": author, "tekst": c})

if not data:
    raise ValueError("❌ Geen tekstfragmenten gevonden. Controleer of je bestanden tekst bevatten met minstens 5 woorden per regel.")

df = pd.DataFrame(data)
print(f"✅ {len(df)} tekstfragmenten geladen van {df['auteur'].nunique()} auteurs: {df['auteur'].unique().tolist()}")

# === 2. Dataset splitsen ===
X_train, X_test, y_train, y_test = train_test_split(
    df["tekst"], df["auteur"], test_size=0.2, random_state=42, stratify=df["auteur"]
)

# === 3. TF-IDF vectorisatie ===
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# === 4. Model trainen ===
model = LogisticRegression(max_iter=2000)
model.fit(X_train_vec, y_train)

# === 5. Evaluatie ===
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"\n📊 Accuracy: {acc*100:.2f}%")
print("\nClassification report:\n", classification_report(y_test, y_pred))

# === 6. Opslaan model ===
joblib.dump(model, MODEL_FILE)
joblib.dump(vectorizer, VECTORIZER_FILE)
print(f"\n💾 Model opgeslagen als {MODEL_FILE} en vectorizer als {VECTORIZER_FILE}")
