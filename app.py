import streamlit as st
import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read FAQ
data = pd.read_csv("faq.csv")

# Stopwords
stop_words = set(stopwords.words("english"))

# Preprocessing function
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [
        word for word in tokens
        if word not in stop_words
        and word not in string.punctuation
    ]
    return " ".join(tokens)

# Process questions
data["Processed"] = data["Question"].apply(preprocess)

# TF-IDF
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(data["Processed"])

# UI
st.title("🤖 FAQ Chatbot")

user_question = st.text_input("Ask your question")

if st.button("Get Answer"):

    user_processed = preprocess(user_question)

    user_vector = vectorizer.transform([user_processed])

    similarity = cosine_similarity(user_vector, faq_vectors)

    best_match = similarity.argmax()

    score = similarity.max()

    if score > 0.3:
        st.success(data.iloc[best_match]["Answer"])
    else:
        st.error("Sorry! I don't know the answer.")