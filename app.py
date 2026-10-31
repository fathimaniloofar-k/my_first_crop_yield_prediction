
import streamlit as st
import pandas as pd
import joblib
import os

# ----------------------------------------
# 1. WEBSITE CONFIGURATION
# ----------------------------------------

st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide"
)

# ----------------------------------------
# 2. LOAD DATASET
# ----------------------------------------

@st.cache_data
def load_dataset():

    df = pd.read_csv("crop_yield.csv")

    return df


# ----------------------------------------
# 3. LOAD TRAINED MODEL
# ----------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load("crop_yield_model.pkl")

    return model


# Load dataset and model
try:

    df = load_dataset()

    model = load_model()

except FileNotFoundError:

    st.error(
        "Dataset or trained model file not found. "
        "Please check your project folder."
    )

    st.stop()


# ----------------------------------------
# 4. WEBSITE HEADER
# ----------------------------------------

st.title("🌾 Agricultural Crop Yield Predictor")

st.write(
    "Predict crop yield using environmental conditions "
    "and a trained Machine Learning model."
)

st.markdown("---")


# ----------------------------------------
# 5. SIDEBAR
# ----------------------------------------

st.sidebar.header("About Project")

st.sidebar.write(
    "This application predicts agricultural crop yield "
    "using Linear Regression."
)

st.sidebar.info(
    "Machine Learning Project"
)


# ----------------------------------------
# 6. INPUT SECTION
# ----------------------------------------

st.header("🛠️ Environmental Inputs")

col1, col2 = st.columns(2)


# ----------------------------------------
# 7. CATEGORICAL INPUTS
# ----------------------------------------

with col1:

    st.subheader("Crop Information")

    region = st.selectbox(
        "Region",
        sorted(df["Region"].dropna().unique())
    )

    soil_type = st.selectbox(
        "Soil Type",
        sorted(df["Soil_Type"].dropna().unique())
    )

    crop = st.selectbox(
        "Crop",
        sorted(df["Crop"].dropna().unique())
    )

    weather = st.selectbox(
        "Weather Condition",
        sorted(df["Weather_Condition"].dropna().unique())
    )


# ----------------------------------------
# 8. NUMERICAL INPUTS
# ----------------------------------------

with col2:

    st.subheader("Environmental Conditions")

    rainfall = st.slider(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=2000.0,
        value=500.0
    )

    temperature = st.slider(
        "Temperature (Celsius)",
        min_value=0.0,
        max_value=50.0,
        value=25.0
    )

    days_to_harvest = st.slider(
        "Days to Harvest",
        min_value=30,
        max_value=365,
        value=120
    )


# ----------------------------------------
# 9. FERTILIZER AND IRRIGATION
# ----------------------------------------

st.markdown("---")

st.subheader("🌱 Farming Practices")

col3, col4 = st.columns(2)

with col3:

    fertilizer = st.toggle(
        "Fertilizer Used",
        value=False
    )

with col4:

    irrigation = st.toggle(
        "Irrigation Used",
        value=False
    )


# ----------------------------------------
# 10. PREDICTION BUTTON
# ----------------------------------------

st.markdown("---")

if st.button(
    "🔮 Predict Crop Yield",
    use_container_width=True
):

    # Create input dataframe
    input_data = pd.DataFrame({

        "Region": [region],

        "Soil_Type": [soil_type],

        "Crop": [crop],

        "Rainfall_mm": [rainfall],

        "Temperature_Celsius": [temperature],

        "Fertilizer_Used": [fertilizer],

        "Irrigation_Used": [irrigation],

        "Weather_Condition": [weather],

        "Days_to_Harvest": [days_to_harvest]

    })


    # ----------------------------------------
    # 11. PREDICT USING TRAINED MODEL
    # ----------------------------------------

    try:

        prediction = model.predict(input_data)

        predicted_yield = prediction[0]


        # ----------------------------------------
        # 12. DISPLAY RESULT
        # ----------------------------------------

        st.success("Prediction Completed Successfully!")

        st.metric(
            label="Estimated Crop Yield",
            value=f"{predicted_yield:.2f} tons/hectare"
        )

        st.write(
            f"Selected Crop: **{crop}**"
        )

        st.write(
            f"Selected Region: **{region}**"
        )


        # ----------------------------------------
        # 13. DISPLAY INPUT SUMMARY
        # ----------------------------------------

        with st.expander("View Input Details"):

            st.dataframe(
                input_data,
                use_container_width=True
            )


    except Exception as e:

        st.error(
            "Prediction failed. Please check whether "
            "the input columns match your trained model."
        )

        st.write(str(e))
