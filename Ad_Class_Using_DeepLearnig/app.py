from flask import Flask, request, render_template
import pickle
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize Flask app
app = Flask(__name__)

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

@app.route('/')
def home():
    return render_template('index.html', result=None)  # Pass None as initial result

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from form
    title = request.form.get('title', '')
    description = request.form.get('description', '')

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

    # Render the result on the same page
    return render_template('index.html', result={"category": predicted_category, "confidence": confidence})

if __name__ == '__main__':
    app.run(debug=True)
