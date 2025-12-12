# 🚀 Next Word Prediction using LSTM & GRU

A lightweight NLP project that predicts the next word in a sentence using deep learning models (LSTM & GRU). The project includes data cleaning, tokenization, model training, and a Streamlit web app for real-time auto-complete suggestions.

---

## 📁 Project Structure

```
next-word-predictor/
│
├── data/
│   └── dataset.txt
│
├── model/
│   ├── lstm_model.keras
│   ├── gru_model.keras
│   └── tokenizer.pkl
│
├── app.py
├── lstm_training.ipynb
├── gru_training.ipynb
└── README.md
```

---

## 🧠 Models Included

### **1️⃣ LSTM-based Next Word Predictor**

* Trained using TensorFlow/Keras
* Uses Embedding + LSTM + Dense
* Predicts next word for any given input text
* Handles unknown words using OOV token

### **2️⃣ GRU-based Next Word Predictor**

* Faster training
* Requires fewer parameters
* Produces competitive accuracy
* Integrated with the same tokenizer for consistency

---

## ⚙️ Tech Stack

| Component         | Technology                       |
| ----------------- | -------------------------------- |
| Model Training    | Python, TensorFlow/Keras         |
| Data Processing   | NumPy, Regex, Tokenizer          |
| Frontend UI       | Streamlit                        |
| Model Storage     | `.keras` (TensorFlow SavedModel) |
| Tokenizer Storage | `.pkl` (Pickle)                  |

---

## 🔧 Training the Models

### **Step 1 — Clean & Preprocess Data**

Both LSTM & GRU notebooks include:

* Lowercasing
* Removing punctuation
* Removing special characters
* Tokenization
* Generating input sequences
* Creating n-grams
* One-hot encoding labels

### **Step 2 — Train LSTM**

```python
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_len-1))
model.add(LSTM(150))
model.add(Dense(vocab_size, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam")
model.fit(X, y, epochs=100, batch_size=32)
model.save("model/lstm_model.keras")
```

### **Step 3 — Train GRU**

```python
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_len-1))
model.add(GRU(150))
model.add(Dense(vocab_size, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam")
model.fit(X, y, epochs=100, batch_size=32)
model.save("model/gru_model.keras")
```

---

## 💾 Saving the Tokenizer

```python
import pickle

with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
```

---

## ▶️ Running the Web App

### **Install Dependencies**

```
pip install -r requirements.txt
```

### **Launch Streamlit**

```
streamlit run app.py
```

### Features

✔ Auto-complete suggestions
✔ Fetch suggestions without clicking a button
✔ Loads LSTM or GRU model dynamically
✔ Clean and minimal UI

---

## 🧪 Model Prediction Logic

```python
def predict_next(word_sequence, top_k=4):
    tokens = tokenizer.texts_to_sequences([word_sequence])[0]
    tokens_padded = pad_sequences([tokens], maxlen=max_len-1, padding="pre")

    prediction = model.predict(tokens_padded)
    top_indices = prediction[0].argsort()[-top_k:][::-1]

    return [index_word[i] for i in top_indices]
```

---

## 📌 Notes

* Ensure `model` folder contains `.keras` and `.pkl` files.
* Use same tokenizer for both LSTM and GRU.
* If file-not-found errors occur, check folder names carefully.
* Large dataset recommended for better predictions.

---

## 🧑‍💻 Author

**Guru**
Student & Developer
Working on ML, DL, and NLP projects.
