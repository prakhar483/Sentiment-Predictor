# 📘 Sentiment Analysis on IMDB Movie Reviews using RNN

This project implements a Recurrent Neural Network (RNN) based sentiment analysis model trained on the IMDB Movie Reviews Dataset.
It classifies movie reviews as Positive or Negative using sequential deep learning architectures such as Simple RNN, GRU, and LSTM.

The repository contains:

RNN_Project_1_Sentiment_Analysis.ipynb — full workflow (preprocessing, training, evaluation, visualization)

imdb_lstm_model.pth — final trained LSTM model

# 📁 Project Folder Structure
# 📦 Sentiment-Analysis-RNN
│
# ├── RNN_Project_1_Sentiment_Analysis.ipynb   # Contains all training steps, preprocessing, evaluation
│
# ├── imdb_lstm_model.pth                      # Saved trained LSTM model
│
# └── README.md                                # Documentation

## 📝 Problem Statement

Build a deep learning model using Recurrent Neural Networks (RNNs) to classify the sentiment of IMDB movie reviews as:

0 → Negative

1 → Positive

The objective is to learn sequential dependencies in text data to improve sentiment classification performance.

## 📂 Dataset Overview

- Source: IMDB Large Movie Review Dataset (Hugging Face datasets)

- Total Samples: 50,000

- Training: 25,000

- Testing: 25,000

- Labels: Binary sentiment

Each sample consists of a complete movie review and its sentiment label.

# 🧠 Approach & Workflow
- ✔ Data Preprocessing

- Tokenization

- Vocabulary building

Padding sequences to uniform length

# ✔ Model Architecture

- Implemented Models: Simple RNN, GRU, LSTM

- Loss Function: CrossEntropy Loss

- Metrics: Accuracy, Precision, Recall, F1-score

- Visualization:

- Accuracy vs Epochs

- Loss vs Epochs

- Confusion Matrix

# ✔ Training Output

- LSTM achieved the best performance: 85.26% validation accuracy

- Other models (Simple RNN, GRU, LSTM + GloVe) achieved ~50%, indicating near-random learning

# 📌 Key Insights & Conclusions

- LSTM outperformed all other architectures by a significant margin.

- The vocabulary was very large (40,133 tokens) due to minimal preprocessing.

- GloVe embeddings performed poorly because of a large vocabulary mismatch (17,000+ missing words).

- Models exhibited unstable learning patterns, suggesting noisy preprocessing.

- Training was limited to 15 epochs, preventing full convergence.

# ⚠️ Technical Limitations

- Minimal preprocessing (HTML tags, stopwords, punctuation retained).

- Very large vocabulary increased sparsity and memory usage.

- Deeper architectures underperformed due to limited epochs.

- Minimal hyperparameter tuning reduced optimization quality.

# 🚀 Future Improvements
## 🔧 Preprocessing Enhancements

- Remove HTML tags

- Apply lemmatization or stemming

- Remove stopwords & punctuation

# 🧠 Vocabulary Optimization

- Increase minimum frequency threshold

- Use subword/BPE tokenization

- Limit vocabulary to top-k most frequent words

# 🏗 Model Enhancements

- Bidirectional GRU/LSTM

- Add attention mechanisms

- Use multi-layer LSTMs with dropout and batch normalization

# 🧪 Training Improvements

- Learning rate scheduling

- Gradient clipping

- More epochs + early stopping

- Use AdamW or RMSprop optimizers

# 📚 Embedding Improvements

- Use domain-specific embeddings

- Improve handling of unknown tokens

- Try contextual embeddings (e.g., BERT, RoBERTa)
