# 📰 Fake News Detection using Transformers (BERT)

This project demonstrates an **end-to-end Natural Language Processing (NLP) workflow** to classify news articles as **REAL** or **FAKE** using **Transformer architecture (BERT)**.  
It focuses on clean implementation, efficient tokenization, GPU-accelerated training, and interactive inference.

---

## 📌 Project Overview

The objective of this project is to detect fake news by learning contextual representations of text using **self-attention mechanisms**.  
A pretrained **BERT (bert-base-uncased)** model is fine-tuned for binary text classification following industry-standard deep learning practices.

---

## 🧠 NLP Pipeline

The project follows these key steps:

1. Import required libraries  
2. Load and preprocess the dataset  
3. Clean and combine text features  
4. Split data into training and validation sets  
5. Tokenize text using BERT tokenizer  
6. Create custom PyTorch Dataset & DataLoader  
7. Fine-tune BERT model  
8. Evaluate model performance  
9. Perform real-time inference using user input  

---

## 📂 Dataset

- **File Name:** `news.csv`
- **Target Variable:** `label`
- **Classes:**
  - `0` → REAL
  - `1` → FAKE
- **Text Features:** `title`, `text`

> ⚠️ Ensure the dataset is available in the project directory before running the notebook.

---

## 🛠️ Libraries Used

```python
pandas
numpy
torch
transformers
scikit-learn

