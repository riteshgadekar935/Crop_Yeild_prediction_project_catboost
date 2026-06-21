import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor

# Load dataset
df = pd.read_csv("yield_df.csv")

# Create target
df["q/hecture"] = df["hg/ha_yield"] / 1000

# Training data
X = df.drop(["hg/ha_yield", "q/hecture", "Unnamed: 0"], axis=1)
y = df["q/hecture"]

# Train model
model = CatBoostRegressor(
    n_estimators=201,
    random_state=42,
    verbose=0
)

model.fit(
    X,
    y,
    cat_features=["Area", "Item"]
)

st.set_page_config(
    page_title="Crop Yield Prediction",
    
)

st.title("Crop Yield Prediction")
st.write("Enter the details below to predict crop yield in Quintal/Hectare")

# Inputs
area = st.selectbox(
    "Country",
    sorted(df["Area"].unique())
)

item = st.selectbox(
    "Crop",
    sorted(df["Item"].unique())
)

year = st.number_input(
    "Year",
    min_value=1990,
    max_value=2035,
    value=2024
)

rainfall = st.number_input(
    "Average Rainfall (mm/year)",
    min_value=0.0,
    value=1000.0
)

pesticides = st.number_input(
    "Pesticides (tonnes)",
    min_value=0.0,
    value=100.0
)

temp = st.number_input(
    "Average Temperature (°C)",
    min_value=-20.0,
    max_value=60.0,
    value=25.0
)

if st.button("Predict Yield"):

    input_data = pd.DataFrame({
        "Area": [area],
        "Item": [item],
        "Year": [year],
        "average_rain_fall_mm_per_year": [rainfall],
        "pesticides_tonnes": [pesticides],
        "avg_temp": [temp]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Yield: {prediction:.2f} Quintal/Hectare"
    )