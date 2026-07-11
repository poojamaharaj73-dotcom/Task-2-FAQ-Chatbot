import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read FAQ file
data = pd.read_csv("faq.csv")

# Load stopwords
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

# Preprocess all FAQ questions
data["Processed"] = data["Question"].apply(preprocess)

# Create TF-IDF vectors
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(data["Processed"])

# Ask user a question
user_question = input("You: ")

# Preprocess user question
user_processed = preprocess(user_question)

# Convert user question into vector
user_vector = vectorizer.transform([user_processed])

# Calculate similarity
similarity = cosine_similarity(user_vector, faq_vectors)

# Find best matching question
best_match = similarity.argmax()

# Get highest similarity score
score = similarity.max()

# Display answer
if score > 0.3:
    print("Bot:", data.iloc[best_match]["Answer"])
else:
    print("Bot: Sorry, I don't know the answer.")