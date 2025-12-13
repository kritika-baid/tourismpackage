import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# ------------------ CONFIG ------------------
MODEL_REPO = "kritika25/tourismmodel"
MODEL_FILE = "best_model.joblib"

st.set_page_config(
    page_title="Tourism Conversion Predictor",
    page_icon="🌍",
    layout="centered"
)

# ------------------ LOAD MODEL ------------------
st.title("🌍 Tourism Package Conversion Prediction")
st.markdown(
    "This app predicts whether a customer is **likely to purchase a tourism package** "
    "based on their profile and interaction details."
)

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE,
    repo_type="model"
)

model = joblib.load(model_path)
FEATURES = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "Gender",
    "MaritalStatus",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "Designation",
    "MonthlyIncome"
]

# ------------------ SIDEBAR INPUT ------------------
st.sidebar.header("🧾 Customer Information")

def user_input():
    data = {}

    data["Age"] = st.sidebar.number_input("Age", 18, 90, 30)
    data["CityTier"] = st.sidebar.selectbox("City Tier", [1, 2, 3])
    data["DurationOfPitch"] = st.sidebar.number_input("Duration Of Pitch", 0, 60, 10)
    data["Gender"] = st.sidebar.selectbox("Gender", ["Male", "Female"])
    data["MaritalStatus"] = st.sidebar.selectbox(
        "Marital Status", ["Single", "Married", "Divorced"]
    )
    data["NumberOfPersonVisiting"] = st.sidebar.number_input(
        "Number Of Persons Visiting", 1, 10, 2
    )
    data["NumberOfFollowups"] = st.sidebar.number_input(
        "Number Of Follow-ups", 0, 10, 2
    )
    data["ProductPitched"] = st.sidebar.selectbox(
        "Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe"]
    )
    data["PreferredPropertyStar"] = st.sidebar.selectbox(
        "Preferred Property Star", [3, 4, 5]
    )
    data["NumberOfTrips"] = st.sidebar.number_input(
        "Number Of Trips", 0, 20, 1
    )
    data["Passport"] = st.sidebar.selectbox("Passport", [0, 1])
    data["PitchSatisfactionScore"] = st.sidebar.selectbox(
        "Pitch Satisfaction Score", [1, 2, 3, 4, 5]
    )
    data["OwnCar"] = st.sidebar.selectbox("Own Car", [0, 1])
    data["Designation"] = st.sidebar.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    data["MonthlyIncome"] = st.sidebar.number_input(
        "Monthly Income", 5000, 300000, 50000
    )

    return pd.DataFrame([data])


    # ------------------ ENCODING ------------------
    data["Gender"] = 1 if data["Gender"] == "Male" else 0
    data["Passport"] = 1 if data["Passport"] == "Yes" else 0
    data["OwnCar"] = 1 if data["OwnCar"] == "Yes" else 0

    # Fill missing model features safely
    final_data = {}
    for col in FEATURES:
        final_data[col] = data.get(col, 0)

    return pd.DataFrame([final_data])

input_df = user_input()

# ------------------ PREDICTION ------------------
st.markdown("---")

if st.button("🔮 Predict Conversion"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.success(
            f"✅ **Customer is likely to purchase the package**\n\n"
            f"📈 Probability: **{probability:.2%}**"
        )
    else:
        st.error(
            f" **Customer is unlikely to purchase the package**\n\n"
            f"📉 Probability: **{probability:.2%}**"
        )

# ------------------ DEBUG VIEW ------------------
with st.expander("🔍 View Model Input Data"):
    st.dataframe(input_df)
