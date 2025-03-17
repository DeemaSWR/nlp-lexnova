import streamlit as st
import numpy as np
import re
import string
import joblib
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =======================
# SETUP STREAMLIT PAGE
# =======================
st.set_page_config(layout="wide")
st.title("📰 Fake News Detection App")
st.write("Enter a news article to classify it as **Fake or Real**.")

# =======================
# LOAD CNN MODEL & TOKENIZER
# =======================



# Use the full absolute path to the new location:
cnn_model = load_model("C:\\Users\\deept\\anaconda3\\envs\\IronhackCamp\\cnn_model.keras")

# Load Tokenizer
tokenizer = joblib.load("C:\\Users\\deept\\anaconda3\\envs\\IronhackCamp\\tokenizer.pkl")

# Load Max Sequence Length
max_sequence_length = joblib.load("C:\\Users\\deept\\anaconda3\\envs\\IronhackCamp\\max_sequence_length.pkl")

# =======================
# TEXT PREPROCESSING FUNCTION
# =======================
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub("\d+", "", text)  # Remove numbers
    return text

# =======================
# STREAMLIT USER INPUT
# =======================

title = st.text_input("Enter the News Title:")
text = st.text_area("Enter the News Content:")

# =======================
# PREDICTION FUNCTION
# =======================

if st.button("Predict"):
    if title and text:
        full_text = title + " " + text  # Combine title & content

        # Clean the input text
        cleaned_text = clean_text(full_text)

        # Convert text to sequences using Tokenizer
        sequence = tokenizer.texts_to_sequences([cleaned_text])

        # Pad the sequence to match the training input size
        padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)

        # Make a prediction using the CNN model
        prediction = cnn_model.predict(padded_sequence)[0][0]
        
        
        # Assign label using threshold
        label = "Real News" if prediction > 0.001 else "Fake News"

        # =======================
        # DISPLAY RESULTS
        # =======================

        st.subheader("📌 Prediction Result")
        st.write(f"📝 **Prediction:** {label}")
        st.write(f"📝 **Precision:** {prediction}")

    else:
        st.warning("⚠️ Please enter both the news title and content.")
