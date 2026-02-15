import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("data/IMDB Dataset.csv")

# Create pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=20000,
        ngram_range=(1,2),
        min_df=2,
        sublinear_tf=True
    )),
    ("clf", LogisticRegression(
        C=4,
        max_iter=2000,
        solver='liblinear'
    ))
])

# Train model
pipeline.fit(df['review'], df['sentiment'])

# Save model
joblib.dump(pipeline, "model/sentiment_model.pkl")

print("Model saved successfully!")
