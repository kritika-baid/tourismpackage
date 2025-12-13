import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# ------------------ CONFIG ------------------
MODEL_REPO = "kritika25/tourismmodel"
MODEL_FILE = "best_model.joblib"
TRAIN_DATA_FILE = "Xtrain.csv"  # local or download from HF

st.set_page_config(page_title="Tourism Conversion Predictor", page_icon="🌍", layout="centered")
st.title("🌍 Tourism Package Conversion Prediction")
st.markdown(
    "This app predicts whether a customer is **likely to purchase a tourism package** "
    "based on their profile and interaction details."
)

# ------------------ LOAD MODEL ------------------
model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, repo_type="model")
model = joblib.load(model_path)

# ------------------ LOAD TRAINING DATA ------------------
Xtrain = pd.read_csv(TRAIN_DATA_FILE)

# ------------------ GET EXPECTED FEATURES ------------------
preprocessor = model.named_steps.get("columntransformer") or model.named_steps.get("preprocessor")
numeric_cols = preprocessor.transformers_[0][2]
categorical_cols = preprocessor.transformers_[1][2]
expected_cols = list(numeric_cols) + list(categorical_cols)

# ------------------ SIDEBAR INPUT ------------------
st.sidebar.header("🧾 Customer Information")
input_data = {}

for col in expected_cols:
    if col in numeric_cols:
        min_val = int(Xtrain[col].min())
        max_val = int(Xtrain[col].max())
        default_val = int(Xtrain[col].median())
        input_data[col] = st.sidebar.number_input(
            col,
            min_value=min_val,
            max_value=max_val,
            value=default_val
        )
    else:
        options = sorted(Xtrain[col].dropna().unique())
        default_val = options[0]
        input_data[col] = st.sidebar.selectbox(col, options, index=0)

input_df = pd.DataFrame([input_data])

# ------------------ PREDICTION ------------------
st.markdown("---")
if st.button("🔮 Predict Conversion"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.success(f"✅ Customer is likely to purchase the package\n📈 Probability: {probability:.2%}")
    else:
        st.error(f"❌ Customer is unlikely to purchase the package\n📉 Probability: {probability:.2%}")

# ------------------ DEBUG VIEW ------------------
with st.expander("🔍 View Model Input Data"):
    st.dataframe(input_df)
