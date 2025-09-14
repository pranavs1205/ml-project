# pylint: disable=invalid-name
# pylint: disable=invalid-name
import streamlit as st
import pandas as pd
import requests
import json

st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🤖",
    layout="wide"
)

st.title("Movie Review Sentiment Analyzer 🎬")
st.markdown("This app predicts whether a movie review is **positive** or **negative** using a Deep Learning model.")

API_URL = "http://127.0.0.1:8000"

st.sidebar.header("Make a Prediction")
input_choice = st.sidebar.radio("Choose input method", ["Single Review", "Upload CSV"])

if input_choice == "Single Review":
    review_text = st.sidebar.text_area("Enter a movie review:", "This movie was absolutely fantastic! The acting was superb and the plot was gripping.")
    if st.sidebar.button("Analyze Sentiment"):
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict_text",
                    json={"text": review_text}
                )
                response.raise_for_status()
                
                prediction_data = response.json()
                sentiment = prediction_data.get("sentiment", "N/A")
                
                st.subheader("Analysis Result")
                if sentiment.lower() == 'positive':
                    st.success(f"Predicted Sentiment: **{sentiment.capitalize()}** 😊")
                else:
                    st.error(f"Predicted Sentiment: **{sentiment.capitalize()}** 😠")

            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the API. Is the backend running? Error: {e}")

else: # Upload CSV
    uploaded_file = st.sidebar.file_uploader("Upload a CSV with a 'review' column", type="csv")
    if st.sidebar.button("Analyze Sentiments from CSV"):
        if uploaded_file is not None:
            with st.spinner("Processing file..."):
                try:
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/csv')}
                    response = requests.post(f"{API_URL}/predict_csv", files=files)
                    response.raise_for_status()

                    predictions = response.json()
                    results_df = pd.DataFrame(predictions)
                    
                    st.subheader("Prediction Results")
                    st.dataframe(results_df)

                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name='sentiment_predictions.csv',
                        mime='text/csv',
                    )

                except requests.exceptions.RequestException as e:
                    st.error(f"API Error. Ensure the CSV has a 'review' column. Details: {e}")