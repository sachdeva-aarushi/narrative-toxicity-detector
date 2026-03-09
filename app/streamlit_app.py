import sys
import os
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.predict import predict_text


st.set_page_config(page_title="Narrative Toxicity Detector",layout="wide")

st.sidebar.title("About")

st.sidebar.write(
"""
This application detects toxic or abusive language in text using a trained NLP model.

Model:
TF-IDF + Logistic Regression

Dataset Sources:
- Jigsaw Toxic Comment Dataset
- Hate Speech Dataset
- GoEmotions Dataset
"""
)

st.sidebar.divider()

st.sidebar.subheader("Example Inputs")

example_1 = st.sidebar.button("Example: Clear Toxic")
example_2 = st.sidebar.button("Example: Neutral Criticism")
example_3 = st.sidebar.button("Example: Harassment")


# ---------- HEADER ----------

st.title("Narrative Toxicity Detector")

st.markdown(
"""
Analyze whether a piece of text contains **toxic or abusive language** using machine learning.
"""
)

st.divider()


# ---------- INPUT SECTION ----------

default_text = ""

if example_1:
    default_text = "You are the dumbest person on this platform"

if example_2:
    default_text = "Your idea is kind of stupid but I see your point"

if example_3:
    default_text = "Nobody cares about your opinion"


user_input = st.text_area(
    "Enter text to analyze",
    value=default_text,
    height=150
)

analyze = st.button("Analyze Text")


st.divider()


# ---------- ANALYSIS ----------

if analyze:

    if user_input.strip() == "":
        st.warning("Please enter text before running analysis.")

    else:

        result = predict_text(user_input)

        prediction = result["prediction"]
        score = result["toxicity_score"]
        percentage = result["toxicity_percentage"]

        st.subheader("Analysis Result")

        # Severity label
        if score < 0.2:
            severity = "Low"
            color = "green"
        elif score < 0.4:
            severity = "Moderate"
            color = "orange"
        else:
            severity = "High"
            color = "red"


        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Prediction",
                value=prediction.upper()
            )

        with col2:
            st.metric(
                label="Toxicity Score",
                value=f"{score:.2f}"
            )

        with col3:
            st.metric(
                label="Confidence",
                value=f"{percentage}%"
            )


        st.write("Toxicity Meter")

        st.progress(score)


        st.write(f"Severity Level: **{severity}**")


        # ---------- Explanation ----------

        with st.expander("How the model works"):

            st.write(
            """
            This model analyzes patterns in language to estimate the probability of toxic or abusive speech.

            Pipeline:
            Text → preprocessing → TF-IDF features → Logistic Regression classifier

            Limitations:
            - May struggle with sarcasm
            - Sensitive to certain keywords
            - Cannot fully understand context
            """
            )


st.divider()

st.caption("Narrative Toxicity Detector | ML Demonstration Project")