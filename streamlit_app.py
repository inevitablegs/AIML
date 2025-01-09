import streamlit as st
import pickle
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the tokenizer and model
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)
model = load_model('video_classification_model.h5')

# Preprocessing function
def pre_process(text):
    """Preprocess input text by normalizing and removing unwanted characters."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters and numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Full light-themed custom CSS
st.markdown(
    """
    <style>
    body {
        background-color: #f1f3f5; /* Light background color */
        color: #212529; /* Dark text for good contrast */
        font-family: 'Arial', sans-serif;
        padding: 20px;
    }
    .title {
        font-size: 3em;
        text-align: center;
        color: #0d6efd; /* Blue title */
        margin-bottom: 30px;
    }
    .subheader {
        font-size: 1.5em;
        margin-top: 20px;
        color: #495057; /* Dark grey */
    }
    .success {
        font-size: 1.5em;
        color: #198754; /* Green for success */
        margin-top: 20px;
    }
    .info {
        font-size: 1.2em;
        color: #0dcaf0; /* Light blue for confidence */
    }
    .stTextInput > div, .stTextArea > div {
        background-color: #ffffff; /* White input fields */
        border: 1px solid #ced4da; /* Light grey border */
        border-radius: 8px;
    }
    .stButton > button {
        background-color: #0d6efd; /* Blue button */
        color: white;
        font-size: 1.2em;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        margin-top: 20px;
    }
    .stButton > button:hover {
        background-color: #004085; /* Darker blue on hover */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">Ad Category Predictor</div>', unsafe_allow_html=True)

# Input section
def input_form():
    st.markdown('<div class="subheader">Enter the details of the ad:</div>', unsafe_allow_html=True)
    title = st.text_input("Ad Title", placeholder="Enter the title of the ad", key="title_input")
    description = st.text_area("Ad Description", placeholder="Enter the description of the ad", key="description_input")
    return title, description

# Main app logic
def predict_category(title, description):
    if title.strip() and description.strip():
        # Combine and preprocess the input
        input_text = pre_process(f"{title} {description}")

        # Tokenize and pad the input
        seq = tokenizer.texts_to_sequences([input_text])
        padded_seq = pad_sequences(seq, maxlen=100)  # Assuming maxlen is 100

        # Predict the category
        prediction = model.predict(padded_seq)
        category_index = np.argmax(prediction)

        # Map the category index to a label
        categories = ['art and music', 'food', 'history', 'manufacturing',
                      'science and technology', 'travel']  # Example categories
        predicted_category = categories[category_index]
        confidence = float(np.max(prediction))

        # Display the result
        st.markdown(
            f'<div class="success">Predicted Category: <strong>{predicted_category}</strong></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="info">Confidence: <strong>{confidence:.2f}</strong></div>',
            unsafe_allow_html=True,
        )
    else:
        st.error("Please enter both title and description.")

# App workflow
title, description = input_form()
if st.button("Predict Category"):
    predict_category(title, description)
    # Option to check another prediction
    if st.button("Check Another One", key="check_another"):
        st.experimental_rerun()
