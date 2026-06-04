import joblib
import re

# 1. Load the serialized vectorizer and model
try:
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    model = joblib.load("ensemble_model.joblib")
except FileNotFoundError:
    print("Error: Serialized model files not found. Please run 'python train.py' first.")
    exit()

# Category mapping (based on the LabelEncoder index)
categories = ['business', 'entertainment', 'politics', 'sport', 'tech']



def predict_article(raw_text):
    # Preprocess and vectorise the input
    cleaned_text = clean_text(raw_text)
    vectorized_text = vectorizer.transform([cleaned_text])
    
    # Predict category and get confidence probabilities
    prediction = model.predict(vectorized_text)[0]
    probabilities = model.predict_proba(vectorized_text)[0]
    
    predicted_category = categories[prediction]
    confidence = probabilities[prediction] * 100
    
    print(f"\nPredicted Category: {predicted_category.upper()} (Confidence: {confidence:.2f}%)")
    print("--- Confidence Breakdown ---")
    for cat, prob in zip(categories, probabilities):
        print(f"  {cat.capitalize()}: {prob*100:.2f}%")

# Example Usage:
if __name__ == "__main__":
    sample_article = """
    Microsoft has announced a new series of artificial intelligence features for its 
    Windows operating system, promising to integrate neural processing units directly 
    into standard laptops. Industry analysts say this could spark a major hardware 
    upgrade cycle across corporate enterprises.
    """
    
    print("Classifying sample article...")
    predict_article(sample_article)
