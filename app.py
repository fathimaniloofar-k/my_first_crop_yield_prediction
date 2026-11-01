

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# ==========================================
# 1. WEBSITE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide"
)


# ==========================================
# 2. PROJECT FILES
# ==========================================

DATASET_PATH = "crop_yield_dataset.csv"

MODEL_PATH = "crop_yield_model.pkl"

PREPROCESSOR_PATH = "crop_yield_preprocessor.pkl"


# ==========================================
# 3. LOAD DATASET
# ==========================================

@st.cache_data
def load_dataset():

    df = pd.read_csv(DATASET_PATH)

    return df


# ==========================================
# 4. LOAD MODEL AND PREPROCESSOR
# ==========================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    return model, preprocessor


# ==========================================
# 5. LOAD PROJECT FILES
# ==========================================

try:

    df = load_dataset()

    model, preprocessor = load_model()

except FileNotFoundError as e:

    st.error(
        "Required project file not found."
    )

    st.write(str(e))

    st.stop()

except Exception as e:

    st.error(
        "Unable to load the trained model."
    )

    st.exception(e)

    st.stop()


# ==========================================
# 6. VALIDATE DATASET
# ==========================================

required_columns = [

    "Region",

    "Soil_Type",

    "Crop",

    "Rainfall_mm",

    "Temperature_Celsius",

    "Fertilizer_Used",

    "Irrigation_Used",

    "Weather_Condition",

    "Days_to_Harvest",

    "Yield_tons_per_hectare"

]


missing_columns = [

    column

    for column in required_columns

    if column not in df.columns

]


if missing_columns:

    st.error(
        f"Dataset is missing columns: {missing_columns}"
    )

    st.stop()


# ==========================================
# 7. WEBSITE HEADER
# ==========================================

st.title(
    "🌾 Agricultural Crop Yield Predictor"
)

st.write(
    "Predict agricultural crop yield using "
    "environmental conditions and farming practices."
)

st.markdown("---")


# ==========================================
# 8. SIDEBAR
# ==========================================

st.sidebar.header(
    "About Project"
)

st.sidebar.write(

    "This application predicts crop yield "
    "using a trained Linear Regression model."

)

st.sidebar.info(
    "Machine Learning Project"
)

st.sidebar.markdown(
    "**Dataset:** Crop Yield Dataset"
)

st.sidebar.markdown(
    "**Algorithm:** Linear Regression"
)


# ==========================================
# 9. INPUT SECTION
# ==========================================

st.header(
    "🛠️ Environmental Inputs"
)

col1, col2 = st.columns(2)


# ==========================================
# 10. CATEGORICAL INPUTS
# ==========================================

with col1:

    st.subheader(
        "Crop Information"
    )

    region_options = sorted(
        df["Region"]
        .dropna()
        .unique()
        .tolist()
    )

    soil_options = sorted(
        df["Soil_Type"]
        .dropna()
        .unique()
        .tolist()
    )

    crop_options = sorted(
        df["Crop"]
        .dropna()
        .unique()
        .tolist()
    )

    weather_options = sorted(
        df["Weather_Condition"]
        .dropna()
        .unique()
        .tolist()
    )

    region = st.selectbox(

        "Region",

        region_options

    )

    soil_type = st.selectbox(

        "Soil Type",

        soil_options

    )

    crop = st.selectbox(

        "Crop",

        crop_options

    )

    weather = st.selectbox(

        "Weather Condition",

        weather_options

    )


# ==========================================
# 11. NUMERICAL INPUTS
# ==========================================

with col2:

    st.subheader(
        "Environmental Conditions"
    )

    # Use dataset values to determine input ranges

    rainfall_min = float(
        df["Rainfall_mm"].min()
    )

    rainfall_max = float(
        df["Rainfall_mm"].max()
    )

    rainfall_default = float(
        df["Rainfall_mm"].median()
    )

    temperature_min = float(
        df["Temperature_Celsius"].min()
    )

    temperature_max = float(
        df["Temperature_Celsius"].max()
    )

    temperature_default = float(
        df["Temperature_Celsius"].median()
    )

    days_min = int(
        df["Days_to_Harvest"].min()
    )

    days_max = int(
        df["Days_to_Harvest"].max()
    )

    days_default = int(
        df["Days_to_Harvest"].median()
    )


    rainfall = st.slider(

        "Rainfall (mm)",

        min_value=rainfall_min,

        max_value=rainfall_max,

        value=rainfall_default

    )


    temperature = st.slider(

        "Temperature (Celsius)",

        min_value=temperature_min,

        max_value=temperature_max,

        value=temperature_default

    )


    days_to_harvest = st.slider(

        "Days to Harvest",

        min_value=days_min,

        max_value=days_max,

        value=days_default

    )


# ==========================================
# 12. FERTILIZER AND IRRIGATION
# ==========================================

st.markdown("---")

st.subheader(
    "🌱 Farming Practices"
)

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


# ==========================================
# 13. PREDICTION BUTTON
# ==========================================

st.markdown("---")


if st.button(

    "🔮 Predict Crop Yield",

    use_container_width=True

):

    try:

        # ----------------------------------
        # CREATE INPUT DATAFRAME
        # ----------------------------------

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


        # ----------------------------------
        # CHECK INPUT COLUMNS
        # ----------------------------------

        expected_columns = [

            "Region",

            "Soil_Type",

            "Crop",

            "Rainfall_mm",

            "Temperature_Celsius",

            "Fertilizer_Used",

            "Irrigation_Used",

            "Weather_Condition",

            "Days_to_Harvest"

        ]


        if list(input_data.columns) != expected_columns:

            st.error(
                "Input columns do not match training columns."
            )

            st.stop()


        # ----------------------------------
        # APPLY TRAINED PREPROCESSOR
        # ----------------------------------

        input_data_scaled = preprocessor.transform(

            input_data

        )


        # ----------------------------------
        # PREDICT CROP YIELD
        # ----------------------------------

        prediction = model.predict(

            input_data_scaled

        )


        predicted_yield = float(

            prediction[0]

        )


        # ----------------------------------
        # DISPLAY RESULT
        # ----------------------------------

        st.success(

            "Prediction Completed Successfully!"

        )


        st.metric(

            label="Estimated Crop Yield",

            value=f"{predicted_yield:.2f} tons/hectare"

        )


        # ----------------------------------
        # DISPLAY SELECTED INFORMATION
        # ----------------------------------

        st.subheader(
            "Prediction Summary"
        )


        st.write(

            f"**Selected Crop:** {crop}"

        )


        st.write(

            f"**Selected Region:** {region}"

        )


        st.write(

            f"**Soil Type:** {soil_type}"

        )


        st.write(

            f"**Weather Condition:** {weather}"

        )


        # ----------------------------------
        # DISPLAY INPUT DETAILS
        # ----------------------------------

        with st.expander(

            "View Input Details"

        ):

            st.dataframe(

                input_data,

                use_container_width=True

            )


        # ----------------------------------
        # DISPLAY MODEL INFORMATION
        # ----------------------------------

        st.caption(

            "Prediction generated using the "
            "trained Linear Regression model."

        )


    except Exception as e:

        st.error(

            "Prediction failed!"

        )

        st.exception(e)
