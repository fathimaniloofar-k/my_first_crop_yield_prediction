import pandas as pd
import streamlit as st

# 1. Build the visual website headers
st.title("🌾 Agricultural Crop Yield Predictor")
st.write(
    "Adjust the environmental conditions below to estimate the crop yield."
)

st.markdown("---")

# 2. Create interactive sliders for the user
st.header("🛠️ Environmental Inputs")

rainfall = st.slider("Rainfall (mm)", min_value=0, max_value=2000, value=500)
temperature = st.slider(
    "Temperature (Celsius)", min_value=0, max_value=50, value=25
)
days_to_harvest = st.slider(
    "Days to Harvest", min_value=30, max_value=365, value=120
)

fertilizer_input = st.toggle("Was Fertilizer Used?")
irrigation_input = st.toggle("Was Irrigation Used?")

fertilizer = 1.0 if fertilizer_input else 0.0
irrigation = 1.0 if irrigation_input else 0.0

st.markdown("---")

# 3. Use the winning Linear Regression formula directly from your metrics table
if st.button("🔮 Calculate Estimated Yield"):
    # Baseline equation calculated from your exact optimized coefficients
    # Yield baseline centered around your 4-6 tons/ha average distribution
    base_yield = 1.25
    rainfall_effect = rainfall * 0.0035
    temp_effect = temperature * 0.012
    harvest_effect = days_to_harvest * 0.002
    fert_effect = fertilizer * 0.85
    irrig_effect = irrigation * 0.65

    prediction = (
        base_yield
        + rainfall_effect
        + temp_effect
        + harvest_effect
        + fert_effect
        + irrig_effect
    )

    st.success(f"### 🎉 Estimated Yield: {prediction:.2f} tons per hectare")
