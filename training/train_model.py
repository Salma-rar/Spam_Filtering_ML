import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# =========================
# Paths
# =========================
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(base_dir, "row_data", "spam.csv")
model_path = os.path.join(base_dir, "model")
os.makedirs(model_path, exist_ok=True)

# =========================
# Load Data
# =========================
df = pd.read_csv(data_path, encoding="latin-1")

# Keep only first 2 columns regardless of their names
df = df.rename(columns={"spamORham": "label", "Message": "message"})
df = df[["label", "message"]]
# Drop nulls
df.dropna(subset=["label", "message"], inplace=True)

print(f"✅ Loaded {len(df)} rows | Spam: {df['label'].value_counts().get('spam', 0)} | Ham: {df['label'].value_counts().get('ham', 0)}")

# =========================
# Features / Labels
# =========================
X = df["message"]
y = df["label"].map({"ham": 0, "spam": 1})

# =========================
# Train / Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# TF-IDF  (consistent with notebook: stop_words + bigrams)
# =========================
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# =========================
# Scaling  (MaxAbsScaler works with sparse matrices)
# =========================
scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train_vec)
X_test_scaled  = scaler.transform(X_test_vec)

# =========================
# Model  (MultinomialNB — consistent with notebook)
# NOTE: MultinomialNB requires non-negative input.
#       MaxAbsScaler preserves sign so values stay in [0,1] for TF-IDF → safe.
# =========================
model = MultinomialNB(alpha=0.1)
model.fit(X_train_scaled, y_train)

# =========================
# Evaluation  (threshold 0.7 — consistent with notebook & app.py)
# =========================
y_probs = model.predict_proba(X_test_scaled)[:, 1]
y_pred  = (y_probs > 0.7).astype(int)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 Accuracy (threshold=0.7): {accuracy * 100:.2f}%")
print("\n" + classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# =========================
# Save Artifacts
# =========================
joblib.dump(model,      os.path.join(model_path, "spam_model.pkl"))
joblib.dump(vectorizer, os.path.join(model_path, "tfidf_vectorizer.pkl"))
joblib.dump(scaler,     os.path.join(model_path, "scaler.pkl"))

print("✅ Training done — 3 files saved to /model/")
