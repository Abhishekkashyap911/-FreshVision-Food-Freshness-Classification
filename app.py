"""FreshVision — Streamlit Web Application for Food Freshness Classification.

Run this application:
    streamlit run app.py
"""

from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

from src.data_preprocessing import CLASS_NAMES, preprocess_image_for_prediction

# --- Page Configuration ---
st.set_page_config(
    page_title="FreshVision — Food Freshness Classifier",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Minimal Styling ---
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.15rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-card-fresh {
        background-color: #ECFDF5;
        border-left: 6px solid #10B981;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .status-card-spoiled {
        background-color: #FEF2F2;
        border-left: 6px solid #EF4444;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-fresh {
        color: #059669;
    }
    .metric-spoiled {
        color: #DC2626;
    }
    .disclaimer-box {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.9rem;
        color: #92400E;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Model Loading with Streamlit Cache ---
MODEL_PATH = Path("models/freshvision_model.keras")


@st.cache_resource(show_spinner=False)
def load_classification_model(model_path: Path):
    """Loads the trained Keras model, caching it in memory."""
    if not model_path.exists():
        return None
    return tf.keras.models.load_model(str(model_path))


# --- Sidebar ---
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1610832958506-aa56368176cf?w=400&auto=format&fit=crop&q=60",
        use_container_width=True,
        caption="AI for Food Quality Assessment",
    )
    st.markdown("### 📌 About FreshVision")
    st.markdown(
        """
        **FreshVision** uses Deep Learning (Transfer Learning with **MobileNetV2**) 
        to detect visual signs of freshness or decay in food images.
        
        - **Architecture:** MobileNetV2 (ImageNet weights)
        - **Input Resolution:** 224 × 224 pixels
        - **Target Categories:** Fresh vs. Spoiled
        - **Framework:** TensorFlow / Keras
        """
    )

    st.markdown("---")
    st.markdown("### ⚙️ Model Status")
    if MODEL_PATH.exists():
        st.success("✅ Model loaded (`freshvision_model.keras`)")
    else:
        st.warning("⚠️ No trained model found in `models/`")
        st.markdown(
            """
            To train the model:
            ```bash
            python train.py
            ```
            Or generate quick demo data:
            ```bash
            python train.py --create-sample-data --epochs 2
            ```
            """
        )

# --- Header Section ---
st.markdown('<div class="main-title">🍎 FreshVision</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">AI-Powered Food Freshness Classification</div>',
    unsafe_allow_html=True,
)

# --- Main Columns: Upload & Results ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 1. Upload Food Image")
    uploaded_file = st.file_uploader(
        "Choose a JPG, JPEG, or PNG image:",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear photograph of a fruit, vegetable, or food item.",
    )

    image_to_predict = None

    if uploaded_file is not None:
        try:
            image_to_predict = Image.open(uploaded_file)
            st.image(
                image_to_predict,
                caption=f"Uploaded Image: {uploaded_file.name}",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Error opening image: {e}")
    else:
        # Check if sample test image exists in dataset/test/
        sample_test_files = list(Path("dataset/test").glob("*/*.[jJ][pP][gG]"))
        if sample_test_files:
            st.info("💡 No file uploaded yet. You can also try a sample from the test set:")
            sample_choice = st.selectbox(
                "Select a sample image from dataset/test:",
                options=["(None)"] + [str(p) for p in sample_test_files[:6]],
            )
            if sample_choice != "(None)":
                image_to_predict = Image.open(sample_choice)
                st.image(image_to_predict, caption=f"Sample: {Path(sample_choice).name}", use_container_width=True)
        else:
            st.info("👆 Please upload a food photo above to see freshness classification.")

with col2:
    st.markdown("### 2. Freshness Analysis")

    if image_to_predict is None:
        st.write("Waiting for an image to be uploaded...")
    else:
        model = load_classification_model(MODEL_PATH)

        if model is None:
            st.error("Model file `models/freshvision_model.keras` not found.")
            st.info(
                "💡 Please train the model first by running:\n"
                "```bash\n"
                "python train.py --create-sample-data --epochs 2\n"
                "```\n"
                "or follow the instructions in the project `README.md`."
            )
        else:
            with st.spinner("Analyzing image features with MobileNetV2..."):
                # Preprocess input image
                processed_array = preprocess_image_for_prediction(image_to_predict)

                # Make prediction
                preds = model.predict(processed_array, verbose=0)[0]
                pred_idx = int(np.argmax(preds))
                predicted_class = CLASS_NAMES[pred_idx].capitalize()
                confidence_score = float(preds[pred_idx]) * 100.0

                fresh_prob = float(preds[0]) * 100.0
                spoiled_prob = float(preds[1]) * 100.0

            # Display prediction card
            is_fresh = predicted_class.lower() == "fresh"
            card_class = "status-card-fresh" if is_fresh else "status-card-spoiled"
            metric_class = "metric-fresh" if is_fresh else "metric-spoiled"
            status_icon = "🌿" if is_fresh else "⚠️"

            st.markdown(
                f"""
                <div class="{card_class}">
                    <div style="font-size: 1rem; color: #4B5563; text-transform: uppercase; letter-spacing: 0.05em;">Classification Result</div>
                    <div class="metric-value {metric_class}">{status_icon} Prediction: {predicted_class}</div>
                    <div style="font-size: 1.15rem; font-weight: 600; color: #1F2937; margin-top: 0.3rem;">
                        Confidence: {confidence_score:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Class Probability Breakdown
            st.markdown("#### Probability Breakdown")
            col_fresh, col_spoiled = st.columns(2)
            with col_fresh:
                st.metric(label="Fresh Probability", value=f"{fresh_prob:.1f}%")
                st.progress(fresh_prob / 100.0)
            with col_spoiled:
                st.metric(label="Spoiled Probability", value=f"{spoiled_prob:.1f}%")
                st.progress(spoiled_prob / 100.0)

            # Analysis Explanation
            st.markdown("#### Observation Summary")
            if is_fresh:
                st.write(
                    "✅ **Fresh Appearance:** The deep learning model observed consistent "
                    "surface pigmentation, standard texture patterns, and healthy optical characteristics "
                    "typical of fresh produce."
                )
            else:
                st.write(
                    "❌ **Spoiled Appearance:** The model observed localized color anomalies, "
                    "surface degradation, or irregular texture variations indicative of microbial decay or over-ripeness."
                )

# --- Educational Warning / Disclaimer ---
st.markdown(
    """
    <div class="disclaimer-box">
        <strong>⚠️ Project Disclaimer:</strong> FreshVision is an educational computer-vision project. 
        Predictions may be incorrect and should not be used as the only method for determining whether food is safe to eat.
        Always verify freshness through physical inspection, smell, standard expiration dates, and food safety protocols.
    </div>
    """,
    unsafe_allow_html=True,
)
