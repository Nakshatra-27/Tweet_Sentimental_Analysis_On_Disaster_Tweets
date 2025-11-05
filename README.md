# 🌪️ Disaster Tweet Classification Using Machine Learning and Deep Learning

## 🧩 Project Overview

The project is about **categorizing of the tweets** into disaster-related or otherwise according to the textual contents.
Twitter is becoming a very instrumental source of real-time information especially during natural disasters.
Nonetheless, it is difficult to distinguish between real disaster-related tweets and the general chatter.

This project is aimed at developing and testing several models - both conventionalMachine Learning, and Deep Learning - to detect tweets, related to disasters, and classify them as such with a high level of accuracy.

---

## 📂 Dataset Source

- **Dataset:** [Kaggle - Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)  
- **Training Data:** 7,613 tweets  
- **Test Data:** 3,263 tweets  

### 🧾 Columns:
| Column | Description |
|:-------|:-------------|
| `id` | Unique identifier for each tweet |
| `text` | Tweet content |
| `keyword` | Keyword present in the tweet |
| `location` | Location of the tweet |
| `target` | Binary label (1 = disaster, 0 = not disaster) |

### 🧹 Data Preprocessing
-Redundition and omission of punctuations.
-Stopword removal and tokenization.
-Lemmatization
-Handling missing values
-ML model vectorization using TF-IDF.
-Streamlined Deep Learning tokenization and padding. 

---

## 🧠 Methods

This project implemented **two Machine Learning models** and **one Deep Learning model** for classification.

### 🔹 Logistic Regression
A linear baseline model which is effective in binary text classification based on TF-IDF features.

- **Vectorizer:** TF-IDF (max_features = 5000, bi-grams included)  
- **Accuracy:** 80.7%  
- **Classification Report:**  
  - Precision: 0.79 / 0.83  
  - Recall: 0.89 / 0.69  
  - F1-Score: 0.84 / 0.75  

---

### 🔹 Naive Bayes (MultinomialNB)
An independent word probabilistic model - simple, but works well with short text.

- **Vectorizer:** TF-IDF  
- **Accuracy:** 80.1%  
- **Classification Report:**  
  - Precision: 0.78 / 0.84  
  - Recall: 0.91 / 0.66  
  - F1-Score: 0.84 / 0.74  

---

### 🔹 Bidirectional LSTM with GloVe Embeddings
Machine learning model with the ability to capture contextual meaning and long-term dependencies in text.

**Architecture:**
-Layering pre-trained GloVe (100D) embeddings.  
-Bidirectional LSTM (128 units)  
-Dense layer (64 units, ReLU)  
-Dropout (0.3-0.4)  
-Sigmoid output layer

**Why this model?**  
The BiLSTM reads the text both forward and backward and it captures relationship in contexts which are between words.  
With GloVe embeddings it is combined to give it semantic richness and enhanced text understanding.
**Performance:**  
- **Accuracy:** 81.8%  
- **Precision:** 0.84 / 0.79  
- **Recall:** 0.85 / 0.78  
- **F1-Score:** 0.84 / 0.78  

---

## 📊 Comparative Results

| Model | Accuracy | Precision (avg) | Recall (avg) | F1-Score (avg) |
|:------|:---------:|:----------------:|:--------------:|:---------------:|
| Logistic Regression | 0.807 | 0.81 | 0.79 | 0.80 |
| Naive Bayes | 0.801 | 0.81 | 0.78 | 0.79 |
| **BiLSTM + GloVe** | **0.820** | **0.82** | **0.81** | **0.81** |

The best overall was the **Bidirectional LSTM using GloVe embeddings** which achieved an accuracy of **82%** and was superior to the traditional models in contextual understanding and overall performance.

---

## ⚙️ Steps to Run the Code

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/disaster-tweet-classification.git
   cd disaster-tweet-classification


## 🧪 Experiments Summary
-Comparing Deep Learning (BiLSTM + GloVe) to Compared Machine Learning (TF-IDF + Logistic Regression, Naive Bayes).  
-Investigated accuracy, precision, recall, and F1-score to evaluate the fair model performance.  
-Illustrated the word distributions and examined the misclassified tweets to get to know model behavior.  
-The BiLSTM classifier model was efficient at learning semantic and contextual patterns not seen in the traditional ML models.

---

##  Conclusion

The Bidirectional LSTM that uses GloVe embeddings had the best accuracy of 81.8 percent of all models.  
It better represented the contextual complexities of tweets as compared to the Logistic Regression and Naive Bayes.  

This project proves that TF-IDF-based ML models are good base lines.  
deep learning models are of better performance in terms of text understanding and classification.

---

## 🔗 References

- [Kaggle Dataset - NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)  
- [Stanford NLP - GloVe Word Embeddings](https://nlp.stanford.edu/projects/glove/)  
- [TensorFlow & Keras Documentation](https://www.tensorflow.org/api_docs)  
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)

