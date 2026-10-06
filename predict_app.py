"""
predict_app.py
Paste a clinical trial paragraph -> get the predicted disease.

Run:  python -m streamlit run predict_app.py

Needs these files in the SAME folder:
  disease_model.pkl, tfidf_vectorizer.pkl, label_encoder.pkl
"""

import re

import joblib
import nltk
import numpy as np
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title="Disease Predictor", page_icon="🩺")


@st.cache_resource
def load_everything():
    for package in ["stopwords", "wordnet", "omw-1.4"]:
        nltk.download(package, quiet=True)
    model = joblib.load("disease_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, tfidf, encoder, set(stopwords.words("english")), WordNetLemmatizer()


def preprocess(text, stop_words, lemmatizer):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)


st.title("🩺 Disease Predictor")
st.write("Paste a clinical trial paragraph below. The model tells you which disease it is about.")

try:
    model, tfidf, encoder, stop_words, lemmatizer = load_everything()
except FileNotFoundError as error:
    st.error(f"File not found: {error}. Keep the .pkl files in the same folder as this app.")
    st.stop()

paragraph = st.text_area("Paragraph", height=220, placeholder="Paste the clinical trial summary here...")

if st.button("Predict disease", type="primary"):
    if len(paragraph.split()) < 5:
        st.warning("Please paste at least 5 words.")
    else:
        cleaned = preprocess(paragraph, stop_words, lemmatizer)
        vector = tfidf.transform([cleaned])

        if vector.nnz == 0:
            st.warning("The model does not recognise any word in this paragraph. Try a medical paragraph.")
        else:
            probabilities = model.predict_proba(vector)[0]
            order = np.argsort(probabilities)[::-1]
            best = order[0]

            st.subheader(f"Predicted disease: {encoder.classes_[best].title()}")
            st.write(f"Confidence: **{probabilities[best] * 100:.1f}%**")

            if probabilities[best] < 0.5:
                st.warning("Low confidence. The paragraph may mention more than one disease.")

            st.write("Top 3 possibilities:")
            for i in order[:3]:
                st.write(f"- {encoder.classes_[i].title()}: {probabilities[i] * 100:.1f}%")
