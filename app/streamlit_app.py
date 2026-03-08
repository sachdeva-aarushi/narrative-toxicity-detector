import streamlit as st

st.set_page_config(page_title="Narrative Toxicity Detector",page_icon="🛡️",layout="centered")
st.title("Narrative Toxicity Detector")
st.write("Enter a sentence below to check whether it contains toxic language.")

user_input = st.text_area("Enter text",height=150)
if st.button("Analyze Text"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        st.write("Processing...")