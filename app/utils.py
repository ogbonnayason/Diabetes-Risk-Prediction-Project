from pathlib import Path

import joblib
import pandas as pd

# app/ directory
BASE_DIR = Path(__file__).resolve().parent
# project root (where data/ and models/ live)
ROOT_DIR = BASE_DIR.parent

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"


def load_pima_model_and_scaler():
    model = joblib.load(MODELS_DIR / "pima_xgb.pkl")
    scaler = joblib.load(MODELS_DIR / "pima_scaler.pkl")
    return model, scaler


def load_nhanes_model_and_scaler():
    model = joblib.load(MODELS_DIR / "nhanes_xgb_tuned.pkl")
    scaler = joblib.load(MODELS_DIR / "nhanes_scaler.pkl")
    return model, scaler


def load_datasets():
    pima = pd.read_csv(DATA_DIR / "pima_clean.csv")
    nhanes = pd.read_csv(DATA_DIR / "nhanes_clean.csv")
    return pima, nhanes


# ---------- NHANES feature construction for prediction ----------

NHANES_FEATURES = [
    "RIAGENDR",     # gender (0/1)
    "RIDAGEYR",     # age
    "INDFMPIR",     # income-to-poverty ratio
    "BMXBMI",       # BMI
    "LBXGH",        # HbA1c
    "LBXGLU",       # fasting glucose
    # race dummies – these must match your nhanes_clean.csv columns
    "race_2.0",
    "race_3.0",
    "race_4.0",
    "race_6.0",
    "race_7.0",
]

# You will later rename these to real groups, e.g. "Non-Hispanic White", etc.
RACE_OPTIONS = {
    "Reference group (baseline)": None,      # all race_* = 0
    "Race 2": "race_2.0",
    "Race 3": "race_3.0",
    "Race 4": "race_4.0",
    "Race 6": "race_6.0",
    "Race 7": "race_7.0",
}


def build_nhanes_feature_df(
    gender: str,
    age: float,
    bmi: float,
    hba1c: float,
    glucose: float,
    income_ratio: float,
    race_label: str,
):
    """
    Build a one-row DataFrame with the correct NHANES feature columns.
    gender: "Male" / "Female"
    race_label: one of the keys in RACE_OPTIONS
    """
    features = {col: 0.0 for col in NHANES_FEATURES}

    # gender: adjust this if your encoding is opposite
    # Here we assume 1 = Male, 0 = Female
    features["RIAGENDR"] = 1.0 if gender == "Male" else 0.0

    features["RIDAGEYR"] = float(age)
    features["INDFMPIR"] = float(income_ratio)
    features["BMXBMI"] = float(bmi)
    features["LBXGH"] = float(hba1c)
    features["LBXGLU"] = float(glucose)

    # reset race dummies
    for rcol in ["race_2.0", "race_3.0", "race_4.0", "race_6.0", "race_7.0"]:
        features[rcol] = 0.0

    selected_dummy = RACE_OPTIONS.get(race_label)
    if selected_dummy is not None:
        features[selected_dummy] = 1.0

    df = pd.DataFrame([features])
    return df
