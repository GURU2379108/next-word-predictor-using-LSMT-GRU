import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------------------------------------
# Load Models / Tokenizer
# ----------------------------------------------------------
@st.cache_resource
def load_model_file(path):
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_tokenizer(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ----------------------------------------------------------
# UI CUSTOM CSS
# ----------------------------------------------------------
st.set_page_config(page_title="Next Word Predictor", layout="centered")

st.markdown("""
<style>

body {
    background: #f9fafb;
}

input[type=text] {
    font-size: 20px !important;
    height: 55px !important;
}

.suggestion-chip {
    padding: 8px 15px;
    background: #eef2ff;
    color: #3730a3;
    border-radius: 20px;
    margin-right: 6px;
    margin-top: 6px;
    display: inline-block;
    cursor: pointer;
    font-size: 16px;
}

.suggestion-chip:hover {
    background: #c7d2fe;
}

.ghost-text {
    font-size: 18px;
    color: #9ca3af;
    margin-top: -10px;
}

.header {
    font-size: 32px;
    font-weight: bold;
    padding: 10px;
    background: linear-gradient(90deg, #4f46e5, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# MAIN HEADER
# ----------------------------------------------------------
st.markdown("<p class='header'>🔮 AI Next-Word Prediction</p>", unsafe_allow_html=True)
st.caption("Powered by LSTM / GRU • Real-time autocomplete • Sentence continuation")

# ----------------------------------------------------------
# MODEL SELECTION
# ----------------------------------------------------------
model_choice = st.radio("Select Model", ["LSTM", "GRU"], horizontal=True)

if model_choice == "LSTM":
    model = load_model_file("models/lstm_model.keras")
else:
    model = load_model_file("models/gru_model.keras")

tokenizer = load_tokenizer("models/tokenizer.pkl")

vocab_size = len(tokenizer.word_index) + 1
MAX_LEN = 20

# ----------------------------------------------------------
# Prediction Functions
# ----------------------------------------------------------
def next_word(text):
    if not text.strip():
        return ""
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=MAX_LEN - 1, padding="pre")

    preds = model.predict(seq, verbose=0)[0]
    idx = np.argmax(preds)

    for w, i in tokenizer.word_index.items():
        if i == idx:
            return w
    return ""

def top_k(text, k=5):
    if not text.strip():
        return []
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=MAX_LEN - 1, padding="pre")

    preds = model.predict(seq, verbose=0)[0]
    top = preds.argsort()[-k:][::-1]

    words = []
    for idx in top:
        for w, i in tokenizer.word_index.items():
            if i == idx:
                words.append(w)
                break
    return words

def continue_sentence(text, length=10):
    for _ in range(length):
        w = next_word(text)
        if not w:
            break
        text += " " + w
    return text

# ----------------------------------------------------------
# INPUT BOX (with ghost autocomplete)
# ----------------------------------------------------------
user_text = st.text_input("Start typing...", key="input_text")

ghost = next_word(user_text)
suggestions = top_k(user_text)

if ghost:
    st.markdown(f"<p class='ghost-text'>✨ {ghost}</p>", unsafe_allow_html=True)

# ----------------------------------------------------------
# Suggestion Chips UI
# ----------------------------------------------------------
if suggestions:
    st.write("**Suggestions:**")
    chip_html = ""
    for s in suggestions:
        chip_html += f"""
        <span class='suggestion-chip' onclick="document.querySelector('input[type=text]').value += ' {s}';">{s}</span>
        """
    st.markdown(chip_html, unsafe_allow_html=True)

# ----------------------------------------------------------
# Sentence Continuation
# ----------------------------------------------------------
if st.button("Continue Sentence → Generate 10 Words"):
    st.subheader("Generated Text:")
    st.write(continue_sentence(user_text, 10))

# ----------------------------------------------------------
# Output Box
# ----------------------------------------------------------
st.text_area("Final output:", value=user_text, height=150)
