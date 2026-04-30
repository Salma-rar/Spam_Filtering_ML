import streamlit as st
import joblib
import os

# paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_dir = os.path.join(base_dir, "model")

@st.cache_resource
def load_assets():
    model = joblib.load(os.path.join(model_dir, "spam_model.pkl"))
    vectorizer = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    return model, vectorizer, scaler

model, vectorizer, scaler = load_assets()

# UI
st.title("📩 Spam Message Classifier")

text = st.text_area("Enter your message")

if st.button("Predict"):

    if text.strip():

        x = vectorizer.transform([text])
        x = scaler.transform(x)

        prob = model.predict_proba(x)[:, 1]

        if prob[0] > 0.7:
            st.error(f"🚨 SPAM ({prob[0]:.2%})")
        else:
            st.success(f"✅ HAM ({prob[0]:.2%})")
    else:
        st.warning("Enter text first")