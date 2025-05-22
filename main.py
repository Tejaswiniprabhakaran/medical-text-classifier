import pandas as pd 
import numpy as np
import re
import string
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef,
    hamming_loss
)

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
data = pd.read_csv("C:\\Users\\Tejaswini\\Downloads\\trainmed.csv", encoding='UTF-8')
print("Initial Data Preview:")
print(data.head())

# Drop unnecessary columns
data = data.drop(columns=['abstract_id', 'line_id', 'line_number', 'total_lines'])
print("\nData after dropping unnecessary columns:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Drop duplicate texts
duplicates = data['abstract_text'].duplicated().sum()
print(f"\nDuplicate abstract_text entries: {duplicates}")
data = data.drop_duplicates(subset=['abstract_text'])

# Fill missing abstract_text with empty string and drop rows with missing targets
data['abstract_text'] = data['abstract_text'].fillna('')
data = data.dropna(subset=['target'])

# Lowercase conversion
data['abstract_text'] = data['abstract_text'].str.lower()

# Remove HTML tags and punctuation/special characters
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

data['abstract_text'] = data['abstract_text'].apply(clean_text)

# Text preprocessing: Lemmatization and stopword removal
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_abstract_text(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    filtered_words = [word for word in lemmatized_words if word not in stop_words]
    return ' '.join(filtered_words)

data['abstract_text'] = data['abstract_text'].apply(clean_abstract_text)

# Encode target labels
label_encoder = LabelEncoder()
data['target_encoded'] = label_encoder.fit_transform(data['target'])

# Split data
X = data['abstract_text']
y = data['target_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTrain-test split completed.")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("TF-IDF vectorization done.")

# Random Forest with GridSearchCV
rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced')
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_tfidf, y_train)

# Best model
rf_classifier = grid_search.best_estimator_
rf_classifier.fit(X_train_tfidf, y_train)
print("Model training completed.")

# Save model and vectorizer
with open('rf_model.pkl', 'wb') as model_file:
    pickle.dump(rf_classifier, model_file)
with open('tfidf_vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)
print("Model and vectorizer saved.")

# Predictions
y_pred = rf_classifier.predict(X_test_tfidf)
y_test_decoded = label_encoder.inverse_transform(y_test)
y_pred_decoded = label_encoder.inverse_transform(y_pred)

# Evaluation
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision (Macro): {precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
print(f"Precision (Weighted): {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"Recall (Macro): {recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
print(f"Recall (Weighted): {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"F1-Score (Macro): {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
print(f"F1-Score (Weighted): {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"Cohen's Kappa: {cohen_kappa_score(y_test, y_pred):.4f}")
print(f"Matthews Correlation Coefficient: {matthews_corrcoef(y_test, y_pred):.4f}")
print(f"Hamming Loss: {hamming_loss(y_test, y_pred):.4f}")

print("\nClassification Report:")
print(classification_report(y_test_decoded, y_pred_decoded, zero_division=0))

# Example prediction using saved model
with open('rf_model.pkl', 'rb') as model_file, open('tfidf_vectorizer.pkl', 'rb') as vec_file:
    loaded_model = pickle.load(model_file)
    loaded_vectorizer = pickle.load(vec_file)

new_input = ["This is an example of medical research abstract."]
new_input_transformed = loaded_vectorizer.transform(new_input)
prediction = loaded_model.predict(new_input_transformed)
predicted_label = label_encoder.inverse_transform(prediction)
print(f"\nPredicted Label for Sample Input: {predicted_label[0]}")





