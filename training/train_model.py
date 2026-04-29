import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression

# =========================
# paths
# =========================
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

data_path = os.path.join(base_dir, "row_data", "spam.csv")
model_path = os.path.join(base_dir, "model")

os.makedirs(model_path, exist_ok=True)

# =========================
# load data (IMPORTANT FIX)
# =========================
df = pd.read_csv(data_path, encoding="latin-1")

# drop useless columns (index column وغيره)
df = df.iloc[:, 1:]   # يشيل أول عمود index

df.columns = ["label", "message"]

# =========================
# features / labels
# =========================
X = df["message"]
y = df["label"].map({"ham": 0, "spam": 1})

# =========================
# TF-IDF
# =========================
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# =========================
# scaling
# =========================
scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X_vec)

# =========================
# model
# =========================
model = LogisticRegression()
model.fit(X_scaled, y)

# =========================
# save
# =========================
joblib.dump(model, os.path.join(model_path, "spam_model.pkl"))
joblib.dump(vectorizer, os.path.join(model_path, "tfidf_vectorizer.pkl"))
joblib.dump(scaler, os.path.join(model_path, "scaler.pkl"))

print("✅ Training done successfully!")