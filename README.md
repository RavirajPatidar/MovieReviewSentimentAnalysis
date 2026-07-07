# 🎬 IMDB Movie Review Sentiment Analysis

## 📌 Project Overview

This project focuses on **Sentiment Analysis** of movie reviews using the **IMDB Movie Review Dataset**. The goal is to classify reviews as **Positive** or **Negative** by applying both traditional Machine Learning techniques and Deep Learning approaches.

The project explores and compares multiple approaches:

1. **TF-IDF + Logistic Regression**
2. **TF-IDF + Random Forest**
3. **Sentence Embeddings + Neural Network**

Through extensive preprocessing, exploratory data analysis, feature engineering, and model evaluation, we identify the most effective approach for sentiment classification.

---

## 🚀 Features

* Data collection using Kaggle API
* Text cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* HTML tag removal
* URL removal
* Lowercasing text
* Special character and punctuation removal
* Stopword removal
* Lemmatization
* TF-IDF Feature Extraction
* Sentence Transformer Embeddings
* Logistic Regression Model
* Random Forest Classifier
* Deep Neural Network
* Confusion Matrix Visualization
* Classification Report Evaluation
* Model Performance Comparison

---

## 📂 Dataset

### IMDB Movie Review Dataset

* Total Reviews: **50,000**
* Classes:

  * Positive
  * Negative

Dataset Source:

IMDB Dataset containing labeled movie reviews for binary sentiment classification.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* BeautifulSoup
* NLTK
* Scikit-Learn
* TensorFlow / Keras
* Sentence Transformers

---

## 📊 Project Workflow

### 1. Dataset Collection

The dataset is downloaded using the Kaggle API and extracted for analysis.

### 2. Data Cleaning

The following preprocessing steps were performed:

#### Duplicate Removal

Duplicate reviews were identified and removed.

#### HTML Tag Removal

BeautifulSoup was used to remove HTML tags from reviews.

#### Text Normalization

* Converted text to lowercase
* Removed URLs
* Removed punctuation
* Removed special characters
* Removed numbers

#### Stopword Removal

Common English stopwords were removed using NLTK.

#### Lemmatization

Words were converted to their base forms:

Examples:

* movies → movie
* liked → like
* running → run

---

## 📈 Exploratory Data Analysis (EDA)

Several analyses were performed:

### Review Length Analysis

* Word Count Distribution
* Character Count Distribution

### Sentiment Distribution

Visualization of positive and negative review counts.

### Frequent Word Analysis

Top words were extracted from:

* Entire Dataset
* Positive Reviews
* Negative Reviews

---

# 🔍 Feature Engineering

## TF-IDF Vectorization

TF-IDF was used to convert text reviews into numerical features.

### Configuration

* Maximum Features: 5000
* N-Grams: Unigrams + Bigrams

```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)
```

---

## Feature Importance Analysis

Top TF-IDF features were extracted and visualized to understand influential terms in the dataset.

---

# 🤖 Machine Learning Models

## 1. Logistic Regression

### Configuration

```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

### Performance

| Metric              | Score |
| ------------------- | ----- |
| Validation Accuracy | 88.4% |
| Test Accuracy       | 88.7% |

### Evaluation

* Classification Report
* Confusion Matrix
* Accuracy Score

---

## 2. Random Forest

### Configuration

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
```

### Performance

| Metric              | Score |
| ------------------- | ----- |
| Validation Accuracy | ~85%  |
| Test Accuracy       | ~85%  |

### Evaluation

* Classification Report
* Confusion Matrix
* Accuracy Score

---

# 🧠 Deep Learning Approach

## Sentence Embeddings + Neural Network

Instead of using TF-IDF features, sentence-level semantic embeddings were generated using:

### Embedding Model

```python
all-MiniLM-L6-v2
```

from Sentence Transformers.

### Embedding Dimension

```text
384 Features
```

---

## Neural Network Architecture

```text
Input Layer (384)

Dense(256, ReLU)
Dropout(0.3)
BatchNormalization

Dense(128, ReLU)
Dropout(0.3)
BatchNormalization

Dense(1, Sigmoid)
```

---

## Training Configuration

### Optimizer

```python
Adam
```

### Loss Function

```python
Binary Crossentropy
```

### Metric

```python
Accuracy
```

### Callback

EarlyStopping

```python
monitor='val_loss'
patience=3
restore_best_weights=True
```

---

## Neural Network Performance

| Metric              | Score |
| ------------------- | ----- |
| Validation Accuracy | 83.3% |
| Test Accuracy       | ~83%  |

### Evaluation

* Accuracy Curves
* Loss Curves
* Classification Report
* Confusion Matrix

---

# 📊 Model Comparison

| Model                               | Validation Accuracy | Test Accuracy |
| ----------------------------------- | ------------------- | ------------- |
| Logistic Regression                 | 88.4%               | 88.7%         |
| Random Forest                       | ~85%                | ~85%          |
| Sentence Embedding + Neural Network | 83.3%               | ~83%          |

---

# 🏆 Best Model

### Logistic Regression + TF-IDF

The Logistic Regression model achieved the highest accuracy and demonstrated excellent generalization performance.

Reasons:

* Simple and efficient
* Fast training
* Strong performance on sparse TF-IDF features
* Better accuracy than Random Forest and Neural Network in this project

---

# 📷 Visualizations

The project includes:

* Sentiment Distribution Plot
* Word Count Analysis
* TF-IDF Feature Importance Plot
* Logistic Regression Confusion Matrix
* Random Forest Confusion Matrix
* Neural Network Confusion Matrix
* Accuracy vs Epoch Plot
* Loss vs Epoch Plot

---

# 🔮 Future Improvements

* BERT Embeddings
* DistilBERT
* RoBERTa
* LSTM Networks
* GRU Networks
* Hyperparameter Optimization
* Ensemble Learning
* Explainable AI (SHAP / LIME)

---

# 📁 Project Structure

```text
MovieReviewSentimentAnalysis/

│
├── Dataset/
│   └── IMDB Dataset.csv
│
├── MovieReviewSentimentAnalysis.ipynb
│
├── README.md
│
└── requirements.txt
```

---

# 👨‍💻 Author

**Raviraj Patidar**

M.Sc. Mathematics and Scientific Computing

Interested in:

* Machine Learning
* Deep Learning
* Natural Language Processing
* Generative AI
* Data Science

---

## ⭐ If you found this project useful, consider giving it a star on GitHub.
