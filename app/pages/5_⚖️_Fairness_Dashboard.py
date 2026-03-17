import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils import load_nhanes_model_and_scaler, load_datasets

st.title("⚖️ Fairness Dashboard – NHANES Model")

st.markdown("""
This dashboard analyses the **fairness** of the tuned XGBoost model across:

- **Gender**  
- **Age groups**  
- **Race / Ethnicity (simplified)**  

We compare:

- **Demographic Parity** → rate of positive predictions  
- **Equal Opportunity** → True Positive Rate (TPR)  
- Also report FPR and PPV for each group.
""")


# ==========================
# 1. Load data, model, predictions
# ==========================

@st.cache_resource
def get_data_and_predictions():
    model, scaler = load_nhanes_model_and_scaler()
    _, nhanes = load_datasets()

    X = nhanes.drop("diabetes_label", axis=1)
    y = nhanes["diabetes_label"]

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    X_df = X.copy()
    return X_df, y, y_pred, y_proba


X_df, y_true_all, y_pred_all, y_proba_all = get_data_and_predictions()


# ==========================
# 2. Helper functions
# ==========================

def compute_group_fairness(y_true, y_pred, y_proba, group_values, group_name="group"):
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
        group_name: group_values
    })

    rows = []
    for g, sub in df.groupby(group_name):
        yt = sub["y_true"].values
        yp = sub["y_pred"].values

        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()

        n = len(sub)
        pos_rate = yt.mean() if n > 0 else np.nan          # actual prevalence
        pred_pos_rate = yp.mean() if n > 0 else np.nan     # demographic parity

        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan  # equal opportunity
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan  # precision

        rows.append({
            group_name: g,
            "n": n,
            "pos_rate": pos_rate,
            "pred_pos_rate": pred_pos_rate,
            "TPR": tpr,
            "FPR": fpr,
            "PPV": ppv
        })

    return pd.DataFrame(rows)


def compute_gaps_vs_reference(fair_df, group_col, ref_group=None):
    df = fair_df.set_index(group_col)

    if ref_group is None:
        # pick majority group
        ref_group = df["n"].idxmax()

    dp_diff = df["pred_pos_rate"] - df.loc[ref_group, "pred_pos_rate"]
    eo_diff = df["TPR"] - df.loc[ref_group, "TPR"]

    gaps = pd.DataFrame({
        "group": df.index,
        "reference": ref_group,
        "DP_diff_vs_ref": dp_diff.values,
        "EO_diff_vs_ref": eo_diff.values
    })
    return gaps


def plot_fairness_heatmap(fair_df, group_col, title):
    metrics_for_heatmap = fair_df.set_index(group_col)[["pred_pos_rate", "TPR", "FPR", "PPV"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(metrics_for_heatmap, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(group_col)
    st.pyplot(fig)
    plt.close(fig)


# ==========================
# 3. Derive group labels
# ==========================

# Gender: assume RIAGENDR is 0/1 (0 = Female, 1 = Male as in preprocessing)
gender_labels = X_df["RIAGENDR"].map({0: "Female", 1: "Male"})

# Age groups
age = X_df["RIDAGEYR"]
age_bins = [18, 40, 60, 80, 120]
age_labels = ["18–39", "40–59", "60–79", "80+"]

age_group = pd.cut(age, bins=age_bins, labels=age_labels, right=False)

# Race groups from one-hot columns
race_cols = [c for c in X_df.columns if c.startswith("race_")]

race_mapping = {
    "race_2.0": "Race 2",
    "race_3.0": "Race 3",
    "race_4.0": "Race 4",
    "race_6.0": "Race 6",
    "race_7.0": "Race 7",
}

def derive_race_group(row):
    if len(race_cols) == 0:
        return "Unknown"

    vals = row[race_cols].values
    if vals.sum() == 0:
        return "Reference"
    idx_max = np.argmax(vals)
    col = race_cols[idx_max]
    return race_mapping.get(col, col)

race_group = X_df.apply(derive_race_group, axis=1)


# ==========================
# 4. Compute fairness tables
# ==========================

gender_fairness = compute_group_fairness(
    y_true_all, y_pred_all, y_proba_all,
    gender_labels,
    group_name="gender"
)

age_fairness = compute_group_fairness(
    y_true_all, y_pred_all, y_proba_all,
    age_group,
    group_name="age_group"
)

race_fairness = compute_group_fairness(
    y_true_all, y_pred_all, y_proba_all,
    race_group,
    group_name="race"
)


tab_gender, tab_age, tab_race = st.tabs(["🚻 Gender", "📅 Age Group", "🌎 Race / Ethnicity"])


# ==========================
# 5. Gender fairness
# ==========================

with tab_gender:
    st.subheader("🚻 Gender Fairness")

    st.write("### Group metrics")
    st.dataframe(gender_fairness)

    st.write("### Heatmap")
    plot_fairness_heatmap(gender_fairness, group_col="gender",
                          title="Gender Fairness Metrics")

    st.write("### Gaps vs Reference (Male)")
    fairness_gaps_gender = compute_gaps_vs_reference(
        gender_fairness,
        group_col="gender",
        ref_group="Male"
    )
    st.dataframe(fairness_gaps_gender)

    st.markdown("""
    - **DP_diff_vs_ref**: difference in predicted positive rate vs reference group.  
    - **EO_diff_vs_ref**: difference in True Positive Rate vs reference group.
    """)


# ==========================
# 6. Age fairness
# ==========================

with tab_age:
    st.subheader("📅 Age Group Fairness")

    st.write("### Group metrics")
    st.dataframe(age_fairness)

    st.write("### Heatmap")
    plot_fairness_heatmap(age_fairness, group_col="age_group",
                          title="Age-Group Fairness Metrics")

    st.write("### Gaps vs Reference (majority group)")
    fairness_gaps_age = compute_gaps_vs_reference(
        age_fairness,
        group_col="age_group",
        ref_group=None  # majority group automatically
    )
    st.dataframe(fairness_gaps_age)


# ==========================
# 7. Race fairness
# ==========================

with tab_race:
    st.subheader("🌎 Race / Ethnicity Fairness")

    st.write("### Group metrics")
    st.dataframe(race_fairness)

    st.write("### Heatmap")
    plot_fairness_heatmap(race_fairness, group_col="race",
                          title="Race / Ethnicity Fairness Metrics")

    st.write("### Gaps vs Reference (majority group)")
    fairness_gaps_race = compute_gaps_vs_reference(
        race_fairness,
        group_col="race",
        ref_group=None
    )
    st.dataframe(fairness_gaps_race)
