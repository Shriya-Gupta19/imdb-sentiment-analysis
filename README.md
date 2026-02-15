# 🎬 IMDb Sentiment Analysis

A Machine Learning project that predicts whether a movie review is **Positive** or **Negative** using Natural Language Processing (NLP).

---

##  Project Overview

This project uses the IMDb 50K Movie Reviews dataset to build a sentiment classification model using:

- Text preprocessing
- TF-IDF Vectorization (with bigrams)
- Logistic Regression
- 5-Fold Cross Validation
- Streamlit Web App Deployment

Final Model Accuracy: **90%**

---

##  Project Structure
``` 
sentiment-analysis/
│
├── data/
│ └── IMDB Dataset.csv
│
├── model/
│ └── sentiment_model.pkl
│
├── train.py
├── app.py
├── requirements.txt
└── README.md 
```


---

##  Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

---

##  Approach

1. Data Cleaning (lowercase, remove special characters)
2. Stopword Removal
3. Lemmatization
4. TF-IDF Feature Extraction (unigrams + bigrams)
5. Logistic Regression Model Training
6. Hyperparameter Tuning
7. Cross Validation

---

##  How to Run the Project

### 1️⃣ Install dependencies

### 2️⃣ Train the model

### 3️⃣ Run the web app

---

##  Model Performance

- Test Accuracy: 90%
- 5-Fold Cross Validation: ~90%
- Balanced Dataset (25K Positive / 25K Negative)

---

##  Future Improvements

- Implement Linear SVM
- Add Deep Learning (LSTM / BERT)
- Add Sentiment Confidence Score
- Deploy Online

---

##  Author

Shriya Gupta




