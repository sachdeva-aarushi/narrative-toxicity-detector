import sys
import os
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.predict import predict_text

st.set_page_config(page_title="Narrative Toxicity Detector",page_icon="🛡️",layout="centered")
st.title("Narrative Toxicity Detector")

st.write("Analyze whether a piece of text contains toxic language using a trained machine learning model.")

user_input = st.text_area("Enter text",height=150)

if st.button("Analyze Text"):

    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")

    else:

        result = predict_text(user_input)

        prediction = result["prediction"]
        score = result["toxicity_score"]
        percentage = result["toxicity_percentage"]

        st.subheader("Analysis Result")
        # color-coded label
        if prediction == "toxic":
            st.error(f"Prediction: {prediction.upper()}")
        else:
            st.success(f"Prediction: {prediction.upper()}")

        st.write(f"Toxicity Score: {score:.2f}")
        st.write(f"Confidence: {percentage}%")
        st.progress(score)