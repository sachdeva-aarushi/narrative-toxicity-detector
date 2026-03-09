import sys
import os
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.predict import predict_text


st.set_page_config(page_title="Narrative Toxicity Detector",layout="centered")

st.title("Narrative Toxicity Detector")

st.markdown(
"""
Detect whether a piece of text contains **toxic or abusive language** using a trained NLP model.
""")

st.divider()

st.subheader("Input Text")

user_input = st.text_area(
    "Enter a sentence to analyze",height=150,placeholder="Example: You are the dumbest person ever")

analyze_button = st.button("Analyze Text")

st.divider()

if analyze_button:

    if user_input.strip() == "":
        st.warning("Please enter text before running analysis.")

    else:

        result = predict_text(user_input)
        prediction = result["prediction"]
        score = result["toxicity_score"]
        percentage = result["toxicity_percentage"]

        st.subheader("Analysis Result")

        if prediction == "toxic":
            st.error(f"Toxic language detected")
        else:
            st.success(f"No toxic language detected")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Toxicity Score",value=f"{score:.2f}")

        with col2:
            st.metric(label="Confidence",value=f"{percentage}%")
        st.write("Toxicity Meter")
        st.progress(score)
st.divider()
st.caption("Model: TF-IDF + Logistic Regression")