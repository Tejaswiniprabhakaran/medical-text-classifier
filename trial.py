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
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, cohen_kappa_score, matthews_corrcoef, hamming_loss
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

# Drop duplicates and handle missing values
data = data.drop_duplicates(subset=['abstract_text'])
data['abstract_text'] = data['abstract_text'].fillna('')
data = data.dropna(subset=['target'])

# Lowercase
data['abstract_text'] = data['abstract_text'].str.lower()

# Clean text
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

data['abstract_text'] = data['abstract_text'].apply(clean_text)

# Lemmatization and stopword removal
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_abstract_text(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    filtered_words = [word for word in lemmatized_words if word not in stop_words]
    return ' '.join(filtered_words)

data['abstract_text'] = data['abstract_text'].apply(clean_abstract_text)

# Label Encoding
label_encoder = LabelEncoder()
data['target_encoded'] = label_encoder.fit_transform(data['target'])

# Label distribution after cleaning
print("\nLabel distribution after preprocessing:")
print(data['target'].value_counts())

# Split dataset
X = data['abstract_text']
y = data['target_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Hyperparameter tuning for Random Forest
rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced')
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    estimator=rf_classifier,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_tfidf, y_train)
rf_classifier = grid_search.best_estimator_
rf_classifier.fit(X_train_tfidf, y_train)
print("\nBest Parameters:")
print(grid_search.best_params_)

# Save model and vectorizer
with open('rf_model.pkl', 'wb') as model_file:
    pickle.dump(rf_classifier, model_file)
with open('tfidf_vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

# Predictions
y_pred = rf_classifier.predict(X_test_tfidf)
y_test_decoded = label_encoder.inverse_transform(y_test)
y_pred_decoded = label_encoder.inverse_transform(y_pred)

# Evaluation
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision (Macro): {precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
print(f"Recall (Macro): {recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
print(f"F1-Score (Macro): {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
print(f"Cohen's Kappa: {cohen_kappa_score(y_test, y_pred):.4f}")
print(f"Matthews Corrcoef: {matthews_corrcoef(y_test, y_pred):.4f}")
print(f"Hamming Loss: {hamming_loss(y_test, y_pred):.4f}")

print("\nClassification Report:")
print(classification_report(y_test_decoded, y_pred_decoded, zero_division=0))

# Sample prediction
with open('rf_model.pkl', 'rb') as model_file, open('tfidf_vectorizer.pkl', 'rb') as vec_file:
    loaded_model = pickle.load(model_file)
    loaded_vectorizer = pickle.load(vec_file)

sample_text = ["This is an example of medical research abstract."]
transformed_sample = loaded_vectorizer.transform(sample_text)
predicted = loaded_model.predict(transformed_sample)
predicted_label = label_encoder.inverse_transform(predicted)
print(f"\nPredicted Label for Sample Input: {predicted_label[0]}")
