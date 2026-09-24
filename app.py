import streamlit as st
import pickle
import pandas as pd


# =========================
# LOAD MODEL AND DATA
# =========================

pipe = pickle.load(open("grid_ridge.best_estimator_.pkl", "rb"))
df = pickle.load(open("df.pkl", "rb"))


# =========================
# TITLE
# =========================

st.title("Laptop Predictor")


# =========================
# INPUTS
# =========================

company = st.selectbox(
    "Brand",
    df["brand"].unique()
)

laptop_name = st.selectbox(
    "Type",
    df["name"].unique()
)

spec_rating = st.selectbox(
    "Specs",
    df["spec_rating"].unique()
)

processor = st.selectbox(
    "Processor",
    df["processor"].unique()
)

cpu = st.selectbox(
    "CPU",
    df["CPU"].unique()
)

ram = st.selectbox(
    "RAM (in GB)",
    sorted(df["Ram"].unique())
)

ram_type = st.selectbox(
    "RAM Type",
    df["Ram_type"].unique()
)

rom = st.selectbox(
    "ROM (in GB)",
    df["ROM"].unique()
)

rom_type = st.selectbox(
    "ROM Type",
    df["ROM_type"].unique()
)

gpu = st.selectbox(
    "GPU",
    df["GPU"].unique()
)

display_size = st.number_input(
    "Display Size (in inches)",
    min_value=10.0,
    max_value=20.0,
    value=15.6,
    step=0.1
)

resolution = st.selectbox(
    "Screen Resolution",
    [
        "1920x1080",
        "1366x768",
        "1600x900",
        "3840x2160",
        "3200x1800",
        "2880x1800",
        "2560x1600",
        "2560x1440",
        "2304x1440"
    ]
)

operating_system = st.selectbox(
    "OS",
    df["OS"].unique()
)

warranty = st.selectbox(
    "Warranty",
    df["warranty"].unique()
)


# =========================
# PREDICT
# =========================

if st.button("Predict Price"):

    # Convert resolution into width and height
    resolution_width = int(resolution.split("x")[0])
    resolution_height = int(resolution.split("x")[1])

    # Create query with EXACTLY the 15 features
    query = pd.DataFrame({
        "brand": [company],
        "name": [laptop_name],
        "spec_rating": [spec_rating],
        "processor": [processor],
        "CPU": [cpu],
        "Ram": [ram],
        "Ram_type": [ram_type],
        "ROM": [rom],
        "ROM_type": [rom_type],
        "GPU": [gpu],
        "display_size": [display_size],
        "resolution_width": [resolution_width],
        "resolution_height": [resolution_height],
        "OS": [operating_system],
        "warranty": [warranty]
    })

    # Make sure feature order exactly matches the trained model
    query = query[pipe.feature_names_in_]

    # Predict price
    prediction = pipe.predict(query)[0]

    # Display prediction directly
    st.success(
        f"The predicted price of this configuration is ₹{int(prediction):,}"
    )