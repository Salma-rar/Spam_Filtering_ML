import streamlit as st
import joblib
import os


base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'spam_model.pkl')
vec_path = os.path.join(base_dir, 'tfidf_vectorizer.pkl')

@st.cache_resource  
def load_assets():
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer

model, vectorizer = load_assets()


st.title(" Spam Message Classifier")
st.write("Welcom  enter the message you want to check:")

user_input = st.text_area("Enter your message here:", height=150)

if st.button("Analyze Message"):
    if user_input.strip() != "":

        data = vectorizer.transform([user_input])
        
        prob = model.predict_proba(data)[:, 1]
        
        if prob > 0.7:
            st.error(f"🚨 This is SPAM! (Probability: {prob[0]:.2%})")
        else:
            st.success(f"✅ This is HAM (Safe). (Probability: {prob[0]:.2%})")
    else:
        st.warning("Please enter a message first!")



