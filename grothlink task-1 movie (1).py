#!/usr/bin/env python
# coding: utf-8

# In[21]:


get_ipython().system('pip install kagglehub')

import kagglehub

# Download latest version
path = kagglehub.dataset_download("rounakbanik/the-movies-dataset")

print("Path to dataset files:", path)

#Install required packages
get_ipython().system('pip install kaggle scikit-learn pandas numpy')

# Import libraries
import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, hamming_loss

# Download the dataset
print("Downloading dataset...")
get_ipython().system('kaggle datasets download -d rounakbanik/the-movies-dataset')

# Check if the file exists before unzipping
zip_file = 'the-movies-dataset.zip'
if os.path.exists(zip_file):
    print(f"Unzipping {zip_file}...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall('movies_dataset')
    print("Extraction complete.")
else:
    raise FileNotFoundError(f"{zip_file} not found. Check Kaggle API authentication.")

# Load and preprocess the dataset
df = pd.read_csv('movies_dataset/movies_metadata.csv', low_memory=False)
df = df[['overview', 'genres']].dropna()

# Parse genres from JSON-like string
def parse_genres(genre_string):
    try:
        genres = ast.literal_eval(genre_string)
        return [genre['name'] for genre in genres]
    except:
        return []

df['genres'] = df['genres'].apply(parse_genres)
df = df[df['genres'].map(len) > 0]

# Prepare features and labels
X = df['overview']
y = df['genres']

# Convert text to TF-IDF vectors
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = tfidf.fit_transform(X)

mlb = MultiLabelBinarizer()
y_binary = mlb.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_binary, test_size=0.2, random_state=42)

# Define classifiers to explore
classifiers = {
    "Logistic Regression": OneVsRestClassifier(LogisticRegression(max_iter=1000)),
    "Random Forest": OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42)),
    "Linear SVM": OneVsRestClassifier(LinearSVC(max_iter=1000, random_state=42))
}

# Train and evaluate each classifier
best_model = None
best_hamming_loss = float('inf')
results = {}

for name, clf in classifiers.items():
    print(f"\nTraining {name}...")
    clf.fit(X_train, y_train)
    
    # Predict probabilities (if available) and adjust threshold
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)
        y_pred = (y_prob >= 0.3).astype(int)  # Adjustable threshold
    else:
        y_pred = clf.predict(X_test)
    
    # Evaluation
    report = classification_report(y_test, y_pred, target_names=mlb.classes_, zero_division=0, output_dict=True)
    hl = hamming_loss(y_test, y_pred)
    results[name] = {"report": report, "hamming_loss": hl}
    
    print(f"{name} - Hamming Loss: {hl:.4f}")
    print(f"Micro F1-Score: {report['micro avg']['f1-score']:.4f}")
    print(classification_report(y_test, y_pred, target_names=mlb.classes_, zero_division=0))
    
    # Track the best model based on Hamming Loss
    if hl < best_hamming_loss:
        best_hamming_loss = hl
        best_model = (name, clf)

# Select the best model
best_name, best_clf = best_model
print(f"\nBest Model: {best_name} (Hamming Loss: {best_hamming_loss:.4f})")

# Feature importance (for Logistic Regression or Linear SVM)
if best_name in ["Logistic Regression", "Linear SVM"]:
    genre_idx = list(mlb.classes_).index('Action')
    coef = best_clf.estimators_[genre_idx].coef_.ravel()
    top_indices = np.argsort(coef)[-10:]
    top_words = [tfidf.get_feature_names_out()[i] for i in top_indices]
    print(f"Top words for 'Action' genre (Feature Importance): {top_words}")

# Misclassification analysis
if hasattr(best_clf, "predict_proba"):
    y_prob = best_clf.predict_proba(X_test)
    y_pred = (y_prob >= 0.3).astype(int)
else:
    y_pred = best_clf.predict(X_test)

misclassified_idx = np.where((y_test != y_pred).any(axis=1))[0][:5]  # First 5 misclassified examples
print("\nMisclassification Insights (First 5 Examples):")
for idx in misclassified_idx:
    true_genres = mlb.inverse_transform(y_test[idx].reshape(1, -1))[0]
    pred_genres = mlb.inverse_transform(y_pred[idx].reshape(1, -1))[0]
    plot = X_test[idx]
    print(f"Plot: {plot[:100]}...")
    print(f"True Genres: {true_genres}")
    print(f"Predicted Genres: {pred_genres}\n")

# Example prediction with the best model
sample_plot = ["A group of astronauts fight aliens on a distant planet."]
sample_tfidf = tfidf.transform(sample_plot)
if hasattr(best_clf, "predict_proba"):
    sample_prob = best_clf.predict_proba(sample_tfidf)
    sample_pred = (sample_prob >= 0.3).astype(int)
else:
    sample_pred = best_clf.predict(sample_tfidf)
predicted_genres = mlb.inverse_transform(sample_pred)
print(f"Predicted genres for sample plot: {predicted_genres}")



