import streamlit as st
import joblib
import os

# =========================
# Paths
# =========================
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_dir = os.path.join(base_dir, "model")

# =========================
# Load Model Assets
# =========================
@st.cache_resource
def load_assets():
    try:
        model      = joblib.load(os.path.join(model_dir, "spam_model.pkl"))
        vectorizer = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
        scaler     = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        return model, vectorizer, scaler
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}\nRun `train_model.py` first.")
        st.stop()

model, vectorizer, scaler = load_assets()

# =========================
# UI
# =========================
st.set_page_config(page_title="Spam Classifier", page_icon="📩")
st.title("📩 Spam Message Classifier")


text = st.text_area("Enter your message", height=150, placeholder="Type a message here...")

THRESHOLD = 0.7

if st.button("🔍 Predict", use_container_width=True):
    if text.strip():
        x_vec    = vectorizer.transform([text])
        x_scaled = scaler.transform(x_vec)
        prob     = model.predict_proba(x_scaled)[:, 1][0]

        st.divider()
        if prob > THRESHOLD:
            st.error(f"🚨 **SPAM** — Confidence: {prob:.2%}")
        else:
            st.success(f"✅ **HAM** — Spam probability: {prob:.2%}")

        # Progress bar for visual feedback
        st.progress(float(prob), text=f"Spam score: {prob:.2%}")
    else:
        st.warning("⚠️ Please enter a message first.")
