# 🚀 BERT Sentiment Analysis Pipeline

An end-to-end **sentiment analysis pipeline** built using **BERT** and PyTorch.
This project processes raw tweet data and classifies sentiment using pretrained embeddings and a custom neural network.

---

## 📌 Features

* Text preprocessing (cleaning tweets)
* Tokenization using BERT tokenizer
* Embedding extraction via pretrained BERT
* Custom classification model (PyTorch)
* Training & inference pipeline

---

## ⚙️ Tech Stack

* Python
* PyTorch
* Transformers (HuggingFace)
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

Tweet sentiment dataset from Kaggle:
https://www.kaggle.com/datasets/krishbaisoya/tweets-sentiment-analysis

---

## 🧠 Model Overview

* Uses pretrained BERT to generate embeddings
* Applies a simple linear classifier on CLS token
* Supports binary sentiment classification (Negative / Positive)

---

## ⚠️ Note

For faster training, reduce dataset size or use GPU.
Full dataset (~1.5M samples) can be slow on local machines.

---

## 📈 Future Improvements

* Fine-tune BERT end-to-end
* Add evaluation metrics (F1-score, confusion matrix)
* Optimize training speed

---

## 👨‍💻 Author

Faizan Imran
