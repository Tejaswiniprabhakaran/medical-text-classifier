import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
    cohen_kappa_score, matthews_corrcoef, hamming_loss
)
from sklearn.preprocessing import LabelEncoder
import pickle

# Read dataset
data = pd.read_csv("C:\\Users\\Tejaswini\\Downloads\\trainmed.csv", encoding='UTF-8')
data = data.drop(columns=['abstract_id', 'line_id', 'line_number', 'total_lines'])

# Check for missing values and clean data
data['abstract_text'] = data['abstract_text'].fillna('').str.lower()
data = data.dropna(subset=['target'])
data['abstract_text'] = data['abstract_text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

# Lemmatization and stopword removal
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_abstract_text(text):
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words]
    return ' '.join(words)

data['abstract_text'] = data['abstract_text'].apply(clean_abstract_text)

# Encode target labels
label_encoder = LabelEncoder()
data['target_encoded'] = label_encoder.fit_transform(data['target'])

# Split data into features and target
X = data['abstract_text']
y = data['target_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_classifier.fit(X_train_tfidf, y_train)

# Evaluate model
y_pred = rf_classifier.predict(X_test_tfidf)

# Decode target labels
y_test_decoded = label_encoder.inverse_transform(y_test)
y_pred_decoded = label_encoder.inverse_transform(y_pred)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
cohen_kappa = cohen_kappa_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
hamming = hamming_loss(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (Macro): {precision_macro:.4f}")
print(f"Recall (Macro): {recall_macro:.4f}")
print(f"F1-Score (Macro): {f1_macro:.4f}")
print(f"Cohen's Kappa: {cohen_kappa:.4f}")
print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
print(f"Hamming Loss: {hamming:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_decoded, y_pred_decoded, zero_division=0))

# Save the model, vectorizer, and label encoder as pickle files
with open('rf_model.pkl', 'wb') as model_file:
    pickle.dump(rf_classifier, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

with open('label_encoder.pkl', 'wb') as encoder_file:
    pickle.dump(label_encoder, encoder_file)

print("Model, vectorizer, and label encoder saved successfully as pickle files.")

# Test the saved model with new data
with open('rf_model.pkl', 'rb') as model_file, open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file, open('label_encoder.pkl', 'rb') as encoder_file:
    rf_classifier_loaded = pickle.load(model_file)
    vectorizer_loaded = pickle.load(vectorizer_file)
    label_encoder_loaded = pickle.load(encoder_file)

new_input_data = ["Sample text for prediction"]
new_input_data_transformed = vectorizer_loaded.transform(new_input_data)
predictions = rf_classifier_loaded.predict(new_input_data_transformed)
predicted_labels = label_encoder_loaded.inverse_transform(predictions)
print(f"Predicted Labels for New Data: {predicted_labels}")




