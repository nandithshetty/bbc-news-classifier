import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z ]', '', text)
    return text

df['Text'] = df['Text'].apply(clean_text)

# Encode labels
le = LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])

X = df['Text']
y = df['Category']

# TF-IDF
vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1,2))
X = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Base models
lr = LogisticRegression(max_iter=1000)
nb = MultinomialNB()
svm = SVC(probability=True)

# Hybrid Ensemble (SOFT VOTING)
ensemble = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('nb', nb),
        ('svm', svm)
    ],
    voting='soft'
)

# Train
ensemble.fit(X_train, y_train)

# Evaluate
y_pred = ensemble.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save models
joblib.dump(ensemble, "ensemble_model.joblib")
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
