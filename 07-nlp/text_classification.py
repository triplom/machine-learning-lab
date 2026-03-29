"""
NLP — Text Classification
==========================
Task: Classify restaurant reviews as Positive (1) or Negative (0).

Pipeline:
  1. Clean text (lowercase, remove punctuation, stopwords, stem)
  2. Bag of Words via CountVectorizer
  3. TF-IDF alternative
  4. Train Naive Bayes classifier
  5. Evaluate

Cross-reference: python_learning/13-nlp for advanced NLP with spaCy.
"""

import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Download stopwords on first run
nltk.download("stopwords", quiet=True)

# ------------------------------------------------------------------
# Synthetic dataset — restaurant reviews
# ------------------------------------------------------------------
reviews_data = {
    "Review": [
        "Wow Loved this place.",
        "Crust is not good.",
        "Not tasty and the texture was just nasty.",
        "Stopped by during the late May bank holiday off Rick Steve recommendation and loved it.",
        "The selection on the menu was great and so were the prices.",
        "Now I am getting angry and I want my damn pho.",
        "Honeslty it didn t taste THAT fresh.",
        "The potatoes were actually a bit clumps.",
        "The fries were great too.",
        "A great touch of seasoning.",
        "Waitress was very rude and unhelpful.",
        "Best pizza in town hands down.",
        "Terrible service and mediocre food.",
        "I love this restaurant so much.",
        "Never coming back here again.",
        "Absolutely delicious food and friendly staff.",
        "Overpriced and underwhelming.",
        "The atmosphere was cozy and the food was fantastic.",
        "My order was wrong and the waiter was dismissive.",
        "Five stars all around!",
    ],
    "Liked": [1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
}
df = pd.DataFrame(reviews_data)

# ------------------------------------------------------------------
# 1. Text cleaning
# ------------------------------------------------------------------
ps = PorterStemmer()
stop_words = set(stopwords.words("english")) - {"not", "no"}  # keep negations

corpus = []
for review in df["Review"]:
    review = re.sub("[^a-zA-Z]", " ", review)  # remove non-letters
    review = review.lower()
    words = review.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    corpus.append(" ".join(words))

print("Sample cleaned reviews:")
for orig, clean in zip(df["Review"][:3], corpus[:3]):
    print(f"  Original: {orig}")
    print(f"  Cleaned:  {clean}\n")

# ------------------------------------------------------------------
# 2a. Bag of Words — CountVectorizer
# ------------------------------------------------------------------
cv = CountVectorizer(max_features=100)
X_bow = cv.fit_transform(corpus).toarray()
y = df["Liked"].values

print(f"BoW feature matrix shape: {X_bow.shape}")
print(f"Top features: {cv.get_feature_names_out()[:10]}")
print()

# ------------------------------------------------------------------
# 2b. TF-IDF
# ------------------------------------------------------------------
tfidf = TfidfVectorizer(max_features=100)
X_tfidf = tfidf.fit_transform(corpus).toarray()


# ------------------------------------------------------------------
# 3. Train / evaluate (Naive Bayes)
# ------------------------------------------------------------------
def evaluate(X, y, label):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"=== {label} ===")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))


evaluate(X_bow, y, "Bag of Words + Naive Bayes")
evaluate(X_tfidf, y, "TF-IDF + Naive Bayes")

# ------------------------------------------------------------------
# 4. Predict a new review
# ------------------------------------------------------------------
new_review = "The food was absolutely amazing and the service was great!"
new_cleaned = " ".join(
    [
        ps.stem(w)
        for w in re.sub("[^a-zA-Z]", " ", new_review).lower().split()
        if w not in stop_words
    ]
)
new_vec = cv.transform([new_cleaned]).toarray()
final_model = MultinomialNB()
final_model.fit(cv.transform(corpus).toarray(), y)
prediction = final_model.predict(new_vec)[0]
print(f"\nNew review: '{new_review}'")
print(f"Prediction: {'POSITIVE' if prediction == 1 else 'NEGATIVE'}")
