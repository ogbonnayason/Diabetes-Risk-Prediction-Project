import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📈 Model Explorer")

st.markdown("""
This page compares the performance of different models trained on:

- **PIMA Indians Diabetes** dataset  
- **NHANES** dataset (multi-cycle, tuned XGBoost is final model)
""")


# ==========================
# 1. Hard-coded metrics from your notebooks
#    (Update these numbers if you rerun experiments)
# ==========================

pima_results = pd.DataFrame([
    {"model": "Logistic Regression", "accuracy": 0.7078, "f1": 0.5455, "auc": 0.8129},
    {"model": "Random Forest",       "accuracy": 0.7338, "f1": 0.5859, "auc": 0.8170},
    {"model": "XGBoost",             "accuracy": 0.7727, "f1": 0.6667, "auc": 0.8189},
    {"model": "ANN (MLP)",           "accuracy": 0.7143, "f1": 0.5686, "auc": 0.8176},
])

nhanes_results_tuned = pd.DataFrame([
    {"model": "Logistic Regression (Tuned)", "accuracy": 0.8843, "f1": 0.6278, "auc": 0.9438},
    {"model": "Random Forest (Tuned)",       "accuracy": 0.9405, "f1": 0.7193, "auc": 0.9440},
    {"model": "XGBoost (Tuned)",             "accuracy": 0.9401, "f1": 0.7190, "auc": 0.9469},
    {"model": "ANN (Tuned)",                 "accuracy": 0.9382, "f1": 0.7038, "auc": 0.9442},
])


tab_pima, tab_nhanes = st.tabs(["PIMA", "NHANES"])


# ==========================
# 2. PIMA tab
# ==========================

with tab_pima:
    st.subheader("PIMA – Model Comparison")

    st.dataframe(pima_results.style.format({"accuracy": "{:.3f}", "f1": "{:.3f}", "auc": "{:.3f}"}))

    metric = st.selectbox("Metric to plot", ["accuracy", "f1", "auc"], key="pima_metric")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=pima_results, x="model", y=metric, ax=ax)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"PIMA – {metric.upper()} by Model")
    plt.xticks(rotation=20)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
    **Observation:** On the PIMA dataset, **XGBoost** has the best balance of Accuracy, F1, and AUC,
    although improvements are modest because the dataset is small and noisy.
    """)


# ==========================
# 3. NHANES tab
# ==========================

with tab_nhanes:
    st.subheader("NHANES – Tuned Models Comparison")

    st.dataframe(nhanes_results_tuned.style.format({"accuracy": "{:.3f}", "f1": "{:.3f}", "auc": "{:.3f}"}))

    metric2 = st.selectbox("Metric to plot", ["accuracy", "f1", "auc"], key="nh_metric")

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.barplot(data=nhanes_results_tuned, x="model", y=metric2, ax=ax2)
    ax2.set_ylim(0.6, 1.0)
    ax2.set_ylabel(metric2.upper())
    ax2.set_title(f"NHANES – {metric2.upper()} by Model")
    plt.xticks(rotation=20)
    st.pyplot(fig2)
    plt.close(fig2)

    st.markdown("""
    **Observation:** All tuned models perform well on NHANES.  
    **Tuned XGBoost** offers the best overall compromise with the **highest AUC**,
    and is therefore selected as the **deployed model** for the app.
    """)
