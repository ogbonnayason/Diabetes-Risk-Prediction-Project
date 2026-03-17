import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings

from utils import (
    load_nhanes_model_and_scaler,
    load_datasets,
    build_nhanes_feature_df,
    RACE_OPTIONS,
)

warnings.filterwarnings("ignore", category=UserWarning)
shap.initjs()

st.title("🔍 Explainability – NHANES XGBoost Model")

st.markdown("""
This page provides **explainable AI (XAI)** for the tuned XGBoost model trained on NHANES.

- **Global explanations** – which features are most important overall  
- **Local explanations** – why the model predicted a certain risk for one person  
- **What-if analysis** – how changing a feature (e.g. BMI or HbA1c) changes the predicted risk
""")


# ==========================
# 1. Load model, scaler, and data
# ==========================

@st.cache_resource
def get_model_scaler_and_data():
    model, scaler = load_nhanes_model_and_scaler()
    _, nhanes = load_datasets()

    X = nhanes.drop("diabetes_label", axis=1)
    y = nhanes["diabetes_label"]

    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    return model, scaler, nhanes, X, X_scaled_df, y


model, scaler, nhanes_df, X_nh, X_nh_scaled_df, y_nh = get_model_scaler_and_data()


# ==========================
# 2. Build SHAP explainer + sample for global plots
# ==========================

@st.cache_resource
def get_shap_explainer_and_sample():
    sample = X_nh_scaled_df.sample(
        n=min(1000, len(X_nh_scaled_df)),
        random_state=42
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    return explainer, shap_values, sample


explainer, shap_values_sample, sample_scaled = get_shap_explainer_and_sample()


tab_global, tab_local, tab_whatif = st.tabs(
    ["🌍 Global Explanations", "👤 Individual Explanation", "⚙ What-if Analysis"]
)


# ==========================
# 3. GLOBAL EXPLANATIONS
# ==========================

with tab_global:
    st.subheader("🌍 Global Feature Importance (NHANES)")

    st.markdown("""
    These plots show which features have the **strongest impact** on the model's predictions,
    averaged over many individuals in the dataset.
    """)

    # Beeswarm plot
    st.markdown("#### SHAP Summary Plot (Beeswarm)")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    shap.summary_plot(
        shap_values_sample,
        sample_scaled,
        show=False,
        plot_type="dot"
    )
    st.pyplot(fig1)
    plt.clf()

    # Bar plot of mean |SHAP|
    st.markdown("#### SHAP Feature Importance (Mean |SHAP|)")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    shap.summary_plot(
        shap_values_sample,
        sample_scaled,
        show=False,
        plot_type="bar"
    )
    st.pyplot(fig2)
    plt.clf()

    st.markdown("""
    **How to read this:**

    - Features at the **top** are most influential overall.  
    - Red points = **high feature values**, blue = low values.  
    - If red points are mostly on the **right**, high values **increase** diabetes risk.  
    - If red points are on the **left**, high values **reduce** risk.
    """)


# ==========================
# 4. LOCAL EXPLANATION (one patient)
# ==========================

with tab_local:
    st.subheader("👤 Explain a Single Prediction")

    st.markdown("""
    Enter details for an individual. The model will:

    1. Predict their **diabetes risk**  
    2. Show a **waterfall plot** explaining which features pushed the prediction up or down
    """)

    with st.form("xai_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"], key="xai_gender")
            age = st.number_input("Age (years)", 18, 120, 50, key="xai_age")
            bmi = st.number_input("BMI", 10.0, 70.0, 30.0, 0.1, key="xai_bmi")

        with col2:
            hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 6.5, 0.1, key="xai_hba1c")
            glucose = st.number_input("Fasting Glucose (mg/dL)", 50.0, 400.0, 130.0, 1.0, key="xai_glu")

        with col3:
            income_ratio = st.number_input(
                "Income-to-Poverty Ratio (INDFMPIR)",
                0.0, 10.0, 2.0, 0.1,
                key="xai_income"
            )
            race_label = st.selectbox(
                "Race / Ethnicity (simplified)",
                list(RACE_OPTIONS.keys()),
                key="xai_race"
            )

        submitted_xai = st.form_submit_button("Explain This Case")

    if submitted_xai:
        # Build feature vector (unscaled)
        X_one = build_nhanes_feature_df(
            gender=gender,
            age=age,
            bmi=bmi,
            hba1c=hba1c,
            glucose=glucose,
            income_ratio=income_ratio,
            race_label=race_label,
        )

        # Scale with training scaler
        X_one_scaled = scaler.transform(X_one.values)
        X_one_scaled_df = pd.DataFrame(X_one_scaled, columns=X_one.columns)

        # Prediction
        proba = model.predict_proba(X_one_scaled_df.values)[:, 1][0]
        pred = int(proba >= 0.5)
        risk_pct = proba * 100

        st.markdown("---")
        st.subheader("Prediction Summary")

        colA, colB = st.columns(2)

        with colA:
            st.metric("Estimated Diabetes Risk", f"{risk_pct:.1f}%")
            if risk_pct < 20:
                risk_level = "Low"
                color = "🟢"
            elif risk_pct < 50:
                risk_level = "Moderate"
                color = "🟡"
            else:
                risk_level = "High"
                color = "🔴"
            st.write(f"**Risk Category:** {color} {risk_level}")

        with colB:
            st.write("**Model Decision (threshold = 0.5):**")
            if pred == 1:
                st.error("Model predicts: **Diabetes / High Risk** (class = 1)")
            else:
                st.success("Model predicts: **No Diabetes / Lower Risk** (class = 0)")

        # SHAP local explanation
        st.markdown("### SHAP Waterfall Plot – Feature Contributions")

        # Get SHAP values for this single instance
        shap_values_one = explainer.shap_values(X_one_scaled_df)[0]

        try:
            # Preferred: modern SHAP waterfall
            expl_obj = shap.Explanation(
                values=shap_values_one,
                base_values=explainer.expected_value,
                data=X_one_scaled_df.iloc[0, :],
                feature_names=X_one_scaled_df.columns,
            )
            fig_w, ax_w = plt.subplots(figsize=(8, 6))
            shap.plots.waterfall(expl_obj, show=False)
            st.pyplot(fig_w)
            plt.clf()
        except Exception:
            try:
                # Fallback: legacy waterfall
                fig_w, ax_w = plt.subplots(figsize=(8, 6))
                shap.plots._waterfall.waterfall_legacy(
                    explainer.expected_value,
                    shap_values_one,
                    feature_names=X_one_scaled_df.columns,
                    show=False
                )
                st.pyplot(fig_w)
                plt.clf()
            except Exception as e2:
                st.warning(f"Could not render waterfall plot due to SHAP version issue: {e2}")

        st.markdown("""
        **How to read this plot:**

        - The grey bar in the middle is the model's **baseline risk** (average over the dataset).  
        - **Red bars** push the risk **higher**; **blue bars** push it **lower**.  
        - The final value on the right is **this person's predicted risk**.
        """)

        st.markdown("### Feature Values Used for This Prediction")
        st.write(X_one)


# ==========================
# 5. WHAT-IF ANALYSIS
# ==========================

with tab_whatif:
    st.subheader("⚙ What-if Analysis")

    st.markdown("""
    Here you can perform a simple **counterfactual-style** analysis:

    1. Define a baseline person  
    2. Choose one feature to change (e.g. BMI or HbA1c)  
    3. See how the predicted diabetes risk changes
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        gender_w = st.selectbox("Gender", ["Male", "Female"], key="wi_gender")
        age_w = st.number_input("Age (years)", 18, 120, 50, key="wi_age")
        bmi_w = st.number_input("BMI", 10.0, 70.0, 30.0, 0.1, key="wi_bmi")

    with col2:
        hba1c_w = st.number_input("HbA1c (%)", 3.0, 15.0, 6.5, 0.1, key="wi_hba1c")
        glucose_w = st.number_input("Fasting Glucose (mg/dL)", 50.0, 400.0, 130.0, 1.0, key="wi_glu")

    with col3:
        income_ratio_w = st.number_input(
            "Income-to-Poverty Ratio (INDFMPIR)",
            0.0, 10.0, 2.0, 0.1,
            key="wi_income"
        )
        race_label_w = st.selectbox(
            "Race / Ethnicity (simplified)",
            list(RACE_OPTIONS.keys()),
            key="wi_race"
        )

    st.markdown("#### Choose a feature to modify")

    feature_to_change = st.selectbox(
        "Feature to change",
        ["BMI", "HbA1c", "Fasting Glucose", "Age", "Income-to-Poverty Ratio"],
        key="wi_feature"
    )

    # Set slider defaults based on current value
    if feature_to_change == "BMI":
        base_val = bmi_w
        min_val, max_val, step = 10.0, 70.0, 0.1
    elif feature_to_change == "HbA1c":
        base_val = hba1c_w
        min_val, max_val, step = 3.0, 15.0, 0.1
    elif feature_to_change == "Fasting Glucose":
        base_val = glucose_w
        min_val, max_val, step = 50.0, 400.0, 1.0
    elif feature_to_change == "Age":
        base_val = age_w
        min_val, max_val, step = 18, 120, 1
    else:  # Income-to-Poverty Ratio
        base_val = income_ratio_w
        min_val, max_val, step = 0.0, 10.0, 0.1

    new_val = st.slider(
        f"New value for {feature_to_change}",
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(base_val),
        step=float(step),
        key="wi_slider",
    )

    if st.button("Run What-if Analysis"):
        # Baseline features
        X_base = build_nhanes_feature_df(
            gender=gender_w,
            age=age_w,
            bmi=bmi_w,
            hba1c=hba1c_w,
            glucose=glucose_w,
            income_ratio=income_ratio_w,
            race_label=race_label_w,
        )

        # Modified features
        X_new = X_base.copy()
        if feature_to_change == "BMI":
            X_new["BMXBMI"] = new_val
        elif feature_to_change == "HbA1c":
            X_new["LBXGH"] = new_val
        elif feature_to_change == "Fasting Glucose":
            X_new["LBXGLU"] = new_val
        elif feature_to_change == "Age":
            X_new["RIDAGEYR"] = new_val
        else:
            X_new["INDFMPIR"] = new_val

        # Scale and predict
        X_base_scaled = scaler.transform(X_base.values)
        X_new_scaled = scaler.transform(X_new.values)

        p_base = model.predict_proba(X_base_scaled)[:, 1][0]
        p_new = model.predict_proba(X_new_scaled)[:, 1][0]

        r_base = p_base * 100
        r_new = p_new * 100
        delta = r_new - r_base

        st.markdown("### Results")

        colb1, colb2, colb3 = st.columns(3)
        with colb1:
            st.metric("Baseline risk", f"{r_base:.1f}%")
        with colb2:
            st.metric("New risk", f"{r_new:.1f}%")
        with colb3:
            st.metric("Change", f"{delta:+.1f} percentage points")

        st.markdown("""
        **Interpretation example (for your report):**

        > If this person's **{feature}** changed from **{old:.2f}** to **{new:.2f}**,  
        > the predicted diabetes risk changed from **{rb:.1f}%** to **{rn:.1f}%**.
        """.format(
            feature=feature_to_change,
            old=base_val,
            new=new_val,
            rb=r_base,
            rn=r_new,
        ))
