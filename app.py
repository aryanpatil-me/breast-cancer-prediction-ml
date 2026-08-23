import streamlit as st
import numpy as np
import joblib
import base64


# ==================================================
# Load Model and Scaler
# ==================================================

model = joblib.load("model/svm_model.pkl")
scaler = joblib.load("model/scaler.pkl")


# ==================================================
# Page Configuration
# ==================================================

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)


# ==================================================
# Background Image
# ==================================================

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


background_image = get_base64_image("assets/background.jpg")


# ==================================================
# Custom CSS
# ==================================================

st.markdown(
f"""
<style>

/* ==============================================
   Main Application
   ============================================== */

.stApp {{
    background-image:
        linear-gradient(
            rgba(231, 241, 251, 0.70),
            rgba(231, 241, 251, 0.70)
        ),
        url("data:image/jpeg;base64,{background_image}");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}


/* ==============================================
   Main Container
   ============================================== */

.block-container {{
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}}


/* ==============================================
   Main Title
   ============================================== */

h1 {{
    color: #173A5E !important;
    font-size: 42px !important;
    font-weight: 800 !important;
    text-align: center;
    margin-bottom: 5px;
}}


/* ==============================================
   Subtitle
   ============================================== */

.subtitle {{
    text-align: center;
    color: #2F78C8;
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 25px;
}}


/* ==============================================
   Section Headings
   ============================================== */

h2 {{
    color: #173A5E !important;
    font-weight: 700 !important;
    margin-top: 30px !important;
}}


/* ==============================================
   Number Input Labels
   ============================================== */

div[data-testid="stNumberInput"] label {{
    color: #173A5E !important;
    font-weight: 600 !important;
}}


/* ==============================================
   Number Inputs
   ============================================== */

div[data-testid="stNumberInput"] input {{
    background-color: #FFFFFF !important;
    border: 1.5px solid #9EC9F3 !important;
    border-radius: 8px !important;
    color: #173A5E !important;
}}


div[data-testid="stNumberInput"] input:focus {{
    border: 2px solid #2F78C8 !important;
    box-shadow: 0 0 0 2px rgba(47, 120, 200, 0.15);
}}


/* ==============================================
   Buttons
   ============================================== */

.stButton > button {{
    width: 100%;
    background-color: #2F78C8 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 20px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    transition: all 0.2s ease;
}}


.stButton > button:hover {{
    background-color: #173A5E !important;
    color: #FFFFFF !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 15px rgba(23, 58, 94, 0.25);
}}


/* ==============================================
   Information Box
   ============================================== */

.info-box {{
    background-color: #E7F1FB;
    color: #173A5E;
    border-left: 5px solid #2F78C8;
    padding: 14px 18px;
    border-radius: 10px;
    font-weight: 500;
    margin: 15px 0;
}}


/* ==============================================
   Warning Box
   ============================================== */

.warning-box {{
    background-color: #FFF4CC;
    color: #173A5E;
    border-left: 5px solid #F0B429;
    padding: 14px 18px;
    border-radius: 10px;
    font-weight: 600;
    margin: 15px 0 25px 0;
}}


/* ==============================================
   Success Message
   Keep Streamlit's original transparent style
   but make the text visible.
   ============================================== */

div[data-testid="stAlert"] p {{
    color: #173A5E !important;
    font-weight: 600 !important;
}}


/* ==============================================
   Quick Test Card
   ============================================== */

.quick-test-card {{
    background-color: rgba(255, 255, 255, 0.96);
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #9EC9F3;
    margin-top: 20px;
    margin-bottom: 20px;
    box-shadow: 0 6px 20px rgba(23, 58, 94, 0.10);
}}


.quick-test-title {{
    color: #173A5E;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 5px;
}}


.quick-test-description {{
    color: #555555;
    font-size: 14px;
}}


/* ==============================================
   Prediction Card
   ============================================== */

.prediction-card {{
    background-color: #FFFFFF;
    border-radius: 18px;
    padding: 30px;
    text-align: center;
    margin-top: 25px;
    box-shadow: 0 8px 25px rgba(23, 58, 94, 0.15);
    border: 2px solid #9EC9F3;
}}


.prediction-title {{
    color: #173A5E;
    font-size: 30px;
    font-weight: 800;
    margin-bottom: 8px;
}}


.prediction-text {{
    color: #2F78C8;
    font-size: 16px;
}}


/* ==============================================
   Divider
   ============================================== */

hr {{
    border: none;
    height: 1px;
    background-color: #9EC9F3;
    margin-top: 30px;
    margin-bottom: 30px;
}}


/* ==============================================
   Footer
   ============================================== */

.footer {{
    text-align: center;
    color: #173A5E;
    font-size: 14px;
    margin-top: 40px;
    padding: 20px;
    border-top: 1px solid #9EC9F3;
}}

</style>
""",
unsafe_allow_html=True
)


# ==================================================
# Header
# ==================================================

st.markdown(
"""
<h1>🩺 Breast Cancer Prediction</h1>

<div class="subtitle">
Machine Learning Based Tumor Classification
</div>
""",
unsafe_allow_html=True
)


# ==================================================
# Information
# ==================================================

st.markdown(
"""
<div class="info-box">
Enter the tumor measurements below to predict whether
the tumor is benign or malignant.
</div>
""",
unsafe_allow_html=True
)


st.markdown(
"""
<div class="warning-box">
⚠️ This application is an educational machine-learning
project and should not be used as a medical diagnosis.
</div>
""",
unsafe_allow_html=True
)


# ==================================================
# Test Samples
# ==================================================

benign_test = [
    13.54, 14.36, 87.46, 566.3, 0.09779,
    0.08129, 0.06664, 0.04781, 0.1885, 0.05766,
    0.2699, 0.7886, 2.058, 23.56, 0.008462,
    0.0146, 0.02387, 0.01315, 0.0198, 0.0023,
    15.11, 19.26, 99.7, 711.2, 0.144,
    0.1773, 0.239, 0.1288, 0.2977, 0.07259
]


malignant_test = [
    17.99, 10.38, 122.8, 1001, 0.1184,
    0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399,
    0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019, 0.1622,
    0.6656, 0.7119, 0.2654, 0.4601, 0.1189
]


# ==================================================
# Feature Keys
# ==================================================

feature_keys = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",

    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",

    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst"
]


# ==================================================
# Load Sample Function
# ==================================================

def load_sample(sample):

    for key, value in zip(feature_keys, sample):
        st.session_state[key] = float(value)


# ==================================================
# Quick Test Card
# ==================================================

st.markdown(
"""
<div class="quick-test-card">
<div class="quick-test-title">🧪 Quick Test</div>
<div class="quick-test-description">
Load a sample from the dataset to quickly test the trained
model without entering all 30 values manually.
</div>
</div>
""",
unsafe_allow_html=True
)


# ==================================================
# Quick Test Buttons
# ==================================================

col1, col2 = st.columns(2)


with col1:

    if st.button(
        "🟢 Load Benign Test Sample",
        use_container_width=True
    ):

        load_sample(benign_test)

        st.success("🟢 Benign test sample loaded.")


with col2:

    if st.button(
        "🔴 Load Malignant Test Sample",
        use_container_width=True
    ):

        load_sample(malignant_test)

        st.success("🔴 Malignant test sample loaded.")


# ==================================================
# Mean Features
# ==================================================

st.header("📊 Mean Features")

col1, col2, col3 = st.columns(3)


with col1:

    radius_mean = st.number_input(
        "Radius Mean",
        min_value=0.0,
        key="radius_mean"
    )

    texture_mean = st.number_input(
        "Texture Mean",
        min_value=0.0,
        key="texture_mean"
    )

    perimeter_mean = st.number_input(
        "Perimeter Mean",
        min_value=0.0,
        key="perimeter_mean"
    )

    area_mean = st.number_input(
        "Area Mean",
        min_value=0.0,
        key="area_mean"
    )


with col2:

    smoothness_mean = st.number_input(
        "Smoothness Mean",
        min_value=0.0,
        key="smoothness_mean"
    )

    compactness_mean = st.number_input(
        "Compactness Mean",
        min_value=0.0,
        key="compactness_mean"
    )

    concavity_mean = st.number_input(
        "Concavity Mean",
        min_value=0.0,
        key="concavity_mean"
    )

    concave_points_mean = st.number_input(
        "Concave Points Mean",
        min_value=0.0,
        key="concave_points_mean"
    )


with col3:

    symmetry_mean = st.number_input(
        "Symmetry Mean",
        min_value=0.0,
        key="symmetry_mean"
    )

    fractal_dimension_mean = st.number_input(
        "Fractal Dimension Mean",
        min_value=0.0,
        key="fractal_dimension_mean"
    )


# ==================================================
# Standard Error Features
# ==================================================

st.header("📐 Standard Error Features")

col1, col2, col3 = st.columns(3)


with col1:

    radius_se = st.number_input(
        "Radius SE",
        min_value=0.0,
        key="radius_se"
    )

    texture_se = st.number_input(
        "Texture SE",
        min_value=0.0,
        key="texture_se"
    )

    perimeter_se = st.number_input(
        "Perimeter SE",
        min_value=0.0,
        key="perimeter_se"
    )

    area_se = st.number_input(
        "Area SE",
        min_value=0.0,
        key="area_se"
    )


with col2:

    smoothness_se = st.number_input(
        "Smoothness SE",
        min_value=0.0,
        key="smoothness_se"
    )

    compactness_se = st.number_input(
        "Compactness SE",
        min_value=0.0,
        key="compactness_se"
    )

    concavity_se = st.number_input(
        "Concavity SE",
        min_value=0.0,
        key="concavity_se"
    )

    concave_points_se = st.number_input(
        "Concave Points SE",
        min_value=0.0,
        key="concave_points_se"
    )


with col3:

    symmetry_se = st.number_input(
        "Symmetry SE",
        min_value=0.0,
        key="symmetry_se"
    )

    fractal_dimension_se = st.number_input(
        "Fractal Dimension SE",
        min_value=0.0,
        key="fractal_dimension_se"
    )


# ==================================================
# Worst Features
# ==================================================

st.header("🔬 Worst Features")

col1, col2, col3 = st.columns(3)


with col1:

    radius_worst = st.number_input(
        "Radius Worst",
        min_value=0.0,
        key="radius_worst"
    )

    texture_worst = st.number_input(
        "Texture Worst",
        min_value=0.0,
        key="texture_worst"
    )

    perimeter_worst = st.number_input(
        "Perimeter Worst",
        min_value=0.0,
        key="perimeter_worst"
    )

    area_worst = st.number_input(
        "Area Worst",
        min_value=0.0,
        key="area_worst"
    )


with col2:

    smoothness_worst = st.number_input(
        "Smoothness Worst",
        min_value=0.0,
        key="smoothness_worst"
    )

    compactness_worst = st.number_input(
        "Compactness Worst",
        min_value=0.0,
        key="compactness_worst"
    )

    concavity_worst = st.number_input(
        "Concavity Worst",
        min_value=0.0,
        key="concavity_worst"
    )

    concave_points_worst = st.number_input(
        "Concave Points Worst",
        min_value=0.0,
        key="concave_points_worst"
    )


with col3:

    symmetry_worst = st.number_input(
        "Symmetry Worst",
        min_value=0.0,
        key="symmetry_worst"
    )

    fractal_dimension_worst = st.number_input(
        "Fractal Dimension Worst",
        min_value=0.0,
        key="fractal_dimension_worst"
    )


# ==================================================
# Prediction
# ==================================================

st.divider()


if st.button(
    "🔍 Predict",
    use_container_width=True
):

    input_data = np.array([
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        smoothness_mean,
        compactness_mean,
        concavity_mean,
        concave_points_mean,
        symmetry_mean,
        fractal_dimension_mean,

        radius_se,
        texture_se,
        perimeter_se,
        area_se,
        smoothness_se,
        compactness_se,
        concavity_se,
        concave_points_se,
        symmetry_se,
        fractal_dimension_se,

        radius_worst,
        texture_worst,
        perimeter_worst,
        area_worst,
        smoothness_worst,
        compactness_worst,
        concavity_worst,
        concave_points_worst,
        symmetry_worst,
        fractal_dimension_worst
    ]).reshape(1, -1)


    # Scale input
    input_scaled = scaler.transform(input_data)


    # Prediction
    prediction = model.predict(input_scaled)[0]


    # Display result
    if prediction == 0:

        st.markdown(
"""
<div class="prediction-card">
<div class="prediction-title">🟢 Benign</div>
<div class="prediction-text">
The model predicts the tumor as benign.
</div>
</div>
""",
unsafe_allow_html=True
        )

    else:

        st.markdown(
"""
<div class="prediction-card">
<div class="prediction-title">🔴 Malignant</div>
<div class="prediction-text">
The model predicts the tumor as malignant.
</div>
</div>
""",
unsafe_allow_html=True
        )


# ==================================================
# Footer
# ==================================================

st.markdown(
"""
<div class="footer">
<strong>Breast Cancer Prediction</strong>
<br>
Machine Learning Project
<br>
Built with Python • Scikit-learn • Streamlit
</div>
""",
unsafe_allow_html=True
)