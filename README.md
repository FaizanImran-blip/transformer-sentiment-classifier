# 🚀 BERT Sentiment Analysis Pipeline

This project is a simple **end-to-end sentiment analysis pipeline** built using BERT and PyTorch.
The goal was to take raw tweet data, clean it, convert it into embeddings using a pretrained model, and then classify sentiment using a custom neural network.

--

## 📌 Features

* Basic text preprocessing (cleaning tweets)
* Tokenization using BERT tokenizer
* Embedding extraction using pretrained BERT
* Simple classification layer (PyTorch)
* Full pipeline from data → training → prediction

---

## ⚙️ Tech Stack

* Python
* PyTorch
* HuggingFace Transformers
* Pandas

---

## 📂 Project Structure

```
main.py        # Full pipeline (data → training → prediction)
main1.py       # BERT embedding extraction
pipe.txt       # Notes / pipeline info
```

---

## ▶️ How to Run

```bash
python main.py
```

---

## 📊 Dataset

I used the tweet sentiment dataset from Kaggle:
https://www.kaggle.com/datasets/krishbaisoya/tweets-sentiment-analysis

---

## 🧠 Model Overview

* BERT is used to generate embeddings from text
* The CLS token is taken as representation
* A simple linear layer is used for classification
* Currently supports binary sentiment (Negative / Positive)

---

## ⚠️ Note

The full dataset is quite large (~1.5M rows), so training can be slow on a local machine.
For testing, it's better to use a smaller sample or run on a GPU.

---

## 📈 Future Improvements

* Fine-tune BERT instead of using frozen embeddings
* Add proper evaluation metrics (F1-score, confusion matrix)
* Improve training efficiency


