import streamlit as st
import pickle
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import entropy
import re

# Load the saved model, vectorizer, and label encoder
with open('rf_model.pkl', 'rb') as model_file, \
     open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file, \
     open('label_encoder.pkl', 'rb') as encoder_file:
    rf_classifier = pickle.load(model_file)
    vectorizer = pickle.load(vectorizer_file)
    label_encoder = pickle.load(encoder_file)

# Initialize session state for metrics tracking and history
if "class_distribution" not in st.session_state:
    st.session_state["class_distribution"] = {}
if "history" not in st.session_state:
    st.session_state["history"] = []

# Streamlit UI
st.title("Medical Data Text Classification Analysis")
st.write("This application classifies input text and provides various evaluation metrics.")

# Input text
input_text = st.text_area("Enter the text(s) for classification (separate by newline):")

if st.button("Classify"):
    if input_text.strip():
        texts = input_text.split('\n')  # Split the input by newline to classify multiple texts
        all_results = []  # To store results of each classification

        for idx, text in enumerate(texts):
            if text.strip():  # Ensure non-empty text
                # Measure start time for latency
                start_time = time.time()

                # Preprocess the input text
                input_transformed = vectorizer.transform([text])

                # Make prediction
                prediction = rf_classifier.predict(input_transformed)
                confidence_scores = rf_classifier.predict_proba(input_transformed)
                predicted_label = label_encoder.inverse_transform(prediction)[0]

                # Measure latency
                latency = time.time() - start_time

                # Update class distribution in session state
                if predicted_label in st.session_state["class_distribution"]:
                    st.session_state["class_distribution"][predicted_label] += 1
                else:
                    st.session_state["class_distribution"][predicted_label] = 1

                # Calculate entropy of predictions
                prediction_entropy = entropy(confidence_scores[0])

                # Store individual result
                result = {
                    "text": text,
                    "predicted_label": predicted_label,
                    "confidence_percentage": confidence_scores[0][prediction[0]] * 100,
                    "latency": latency,
                    "entropy": prediction_entropy,
                    "top_n_predictions": []
                }

                # Display top-N predictions
                top_n = 5  # Set N for Top-N predictions
                top_n_indices = np.argsort(confidence_scores[0])[::-1][:top_n]
                top_n_labels = label_encoder.inverse_transform(top_n_indices)
                top_n_confidences = confidence_scores[0][top_n_indices]

                for i, (label, confidence) in enumerate(zip(top_n_labels, top_n_confidences)):
                    result["top_n_predictions"].append((label, confidence * 100))

                all_results.append(result)

        # Display all results
        for result in all_results:
            st.subheader(f"Result for Text {all_results.index(result) + 1}:")
            st.write(f"**Predicted Classification Label:** {result['predicted_label']}")
            st.write(f"**Confidence Score:** {result['confidence_percentage']:.2f}%")
            st.write(f"**Latency:** {result['latency']:.4f} seconds")
            st.write(f"**Prediction Entropy:** {result['entropy']:.4f}")

            st.subheader("Top-N Predictions (with Confidence Scores):")
            for i, (label, confidence) in enumerate(result["top_n_predictions"]):
                st.write(f"Top {i + 1}: {label} ({confidence:.2f}%)")

            # Display heatmap for top-N predictions
            st.subheader("Top-N Predictions Confidence Heatmap:")
            heatmap_data = np.array([confidence for _, confidence in result["top_n_predictions"]]).reshape(1, -1)
            fig, ax = plt.subplots(figsize=(10, 2))  # Adjust figure size for better visualization
            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=".2f",
                xticklabels=[label for label, _ in result["top_n_predictions"]],
                yticklabels=["Confidence"],
                cmap="YlGnBu",
                cbar=False,
                ax=ax
            )
            plt.xlabel("Predicted Categories")
            plt.ylabel("Confidence")
            st.pyplot(fig)

            # Handle out-of-vocabulary terms
            st.subheader("Out-of-Vocabulary (OOV) Rate:")

            # Enhanced tokenization (remove punctuation and split by spaces)
            input_tokens = set(re.findall(r'\b\w+\b', result['text'].lower()))  # Tokenize while ignoring case and punctuation
            training_vocab = set(vectorizer.get_feature_names_out())

            # Identify OOV tokens
            oov_tokens = input_tokens - training_vocab
            oov_rate = len(oov_tokens) / len(input_tokens) if input_tokens else 0

            st.write(f"OOV Rate: {oov_rate:.2%}")
            if oov_tokens:
                st.write(f"OOV Tokens: {', '.join(oov_tokens)}")

        # Add this session's results to history
        st.session_state["history"].append(all_results)

        # Display prediction count distribution
        st.subheader("Prediction Count Distribution:")
        class_labels = list(st.session_state["class_distribution"].keys())
        class_counts = list(st.session_state["class_distribution"].values())

        # Bar chart for distribution
        fig, ax = plt.subplots()
        ax.bar(class_labels, class_counts, color='skyblue')
        plt.xlabel("Class Labels")
        plt.ylabel("Count")
        plt.title("Prediction Count Distribution")
        st.pyplot(fig)

        # Display label counts below the bar chart
        st.write("### Label Count Summary:")
        for label, count in zip(class_labels, class_counts):
            st.write(f"{label}: {count} times")

    else:
        st.warning("Please enter some text for classification.")

# Display classification history
st.sidebar.subheader("History of Classifications")
for i, result_set in enumerate(st.session_state["history"]):
    st.sidebar.write(f"--- History {i + 1} ---")
    for result in result_set:
        st.sidebar.write(f"Predicted Label: {result['predicted_label']} - Confidence: {result['confidence_percentage']:.2f}%")
