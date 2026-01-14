import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load data
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

fake_df["label"] = 1
true_df["label"] = 0

df = pd.concat([fake_df, true_df])
df = df.sample(frac=1).reset_index(drop=True)

df["content"] = df["title"] + " " + df["text"]

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    return text

df["content"] = df["content"].apply(clean_text)

X = df["content"]
y = df["label"]

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

# ---------------- UI ----------------
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detection using Machine Learning")
st.write("Paste a news article below to check whether it is **Fake or Real**.")

user_input = st.text_area("Enter News Text", height=200)

if st.button("Check News"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        vec_input = vectorizer.transform([cleaned])
        prediction = model.predict(vec_input)[0]
        probability = model.predict_proba(vec_input).max()

        if prediction == 1:
            st.error(f"⚠️ Fake News Detected (Confidence: {probability*100:.2f}%)")
        else:
            st.success(f"✅ Real News Detected (Confidence: {probability*100:.2f}%)")
    else:
        st.warning("Please enter some news text.")
