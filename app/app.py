import streamlit as st

st.set_page_config(
    page_title="Explainable & Fair AI for Type 2 Diabetes Risk",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Explainable & Fair AI for Type 2 Diabetes Risk")

st.markdown("""
This web app is part of an MSc project:

**"Explainable and Fair Machine Learning Framework for Predicting Type 2 Diabetes Risk."**

Use the sidebar to navigate:

- **Dataset Explorer** – inspect PIMA & NHANES datasets  
- **Risk Prediction** – enter patient details and get risk score  
- **Explainability (XAI)** – see SHAP-based explanations  
- **Fairness Dashboard** – model behaviour across demographic groups
""")

st.info(
    "This tool is for research and educational purposes only and "
    "must not be used for real clinical decision-making."
)
