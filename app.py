import streamlit as st
import joblib

# Page config
st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# Load model
model = joblib.load("model/sentiment_model.pkl")

# Custom styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎬 IMDb Movie Review Sentiment Analyzer")
st.write("Enter a movie review below to predict whether it is Positive or Negative.")

review = st.text_area("✍️ Write your review here:", height=150)

if st.button("🔍 Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        prediction = model.predict([review])[0]

        if prediction == "positive":
            st.success("😊 Positive Review")
        else:
            st.error("😡 Negative Review")

st.markdown("---")
st.caption("Built using TF-IDF + Logistic Regression | Accuracy: 90%")
