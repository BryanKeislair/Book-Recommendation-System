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
# Verwacht structuur:
# authors/
#   ├── tolkien.txt
#   ├── rowling.txt
#   ├── austen.txt
#   ├── poe.txt
# Elke .txt bevat enkele alinea’s van die auteur

data = []
for file in os.listdir(DATA_DIR):
    if file.endswith(".txt"):
        author = file.replace(".txt", "")
        with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
            text = f.read()
            # Tekst opsplitsen in stukken (bijv. 300 woorden)
            chunks = text.split("\n\n")
            for c in chunks:
                if len(c.strip().split()) > 30:  # minimaal 30 woorden
                    data.append({"auteur": author, "tekst": c.strip()})

df = pd.DataFrame(data)
print(f"✅ {len(df)} tekstfragmenten geladen van {df['auteur'].nunique()} auteurs.")

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
