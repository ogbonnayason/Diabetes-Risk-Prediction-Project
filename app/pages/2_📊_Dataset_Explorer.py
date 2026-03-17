import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_datasets

st.title("📊 Dataset Explorer")

pima, nhanes = load_datasets()

tab1, tab2 = st.tabs(["PIMA", "NHANES"])

with tab1:
    st.subheader("PIMA Dataset")
    st.write(pima.head())
    st.write("Shape:", pima.shape)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Outcome distribution")
        st.bar_chart(pima["Outcome"].value_counts())

    with col2:
        st.write("BMI distribution")
        fig, ax = plt.subplots()
        sns.histplot(pima["BMI"], kde=True, ax=ax)
        st.pyplot(fig)

with tab2:
    st.subheader("NHANES Dataset")
    st.write(nhanes.head())
    st.write("Shape:", nhanes.shape)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Diabetes label distribution")
        st.bar_chart(nhanes["diabetes_label"].value_counts())

    with col2:
        st.write("BMI distribution")
        fig, ax = plt.subplots()
        sns.histplot(nhanes["BMXBMI"], kde=True, ax=ax)
        st.pyplot(fig)
