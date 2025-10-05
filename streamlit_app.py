import tensorflow 
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# ------------------------------
# 1️⃣ Load Pre-trained Model
# ------------------------------
@st.cache_resource
def load_cnn_model():
    model = load_model("mnist_cnn_model.h5")
    return model

model = load_cnn_model()

st.set_page_config(page_title="MNIST Handwriting Recognition", layout="centered")

st.title("🖊️ Handwritten Digit Recognition (MNIST)")
st.markdown("""
Draw a digit (0–9) below and let the trained CNN predict what it is!  
""")

# ------------------------------
# 2️⃣ Drawing Canvas
# ------------------------------
st.subheader("✏️ Draw Your Digit Below")

canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# ------------------------------
# 3️⃣ Predict Button
# ------------------------------
if st.button("🔍 Predict"):
    if canvas_result.image_data is not None:
        # Convert canvas image to 28x28 grayscale
        img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Normalize and reshape
        gray = gray.reshape(1, 28, 28, 1) / 255.0

        # Make prediction
        pred = model.predict(gray)
        predicted_digit = np.argmax(pred)
        confidence = np.max(pred) * 100

        # Display results
        st.subheader("📊 Prediction Results")
        col1, col2 = st.columns(2)

        with col1:
            st.image(canvas_result.image_data, caption="Your Drawing", width=150)

        with col2:
            st.markdown(f"**Predicted Digit:** `{predicted_digit}`")
            st.markdown(f"**Confidence:** `{confidence:.2f}%`")

        # Explain inference concept briefly
        st.markdown("""
        ---
        **Theory Recap:**
        - CNN uses convolution layers to extract features (edges, curves, shapes).
        - Softmax layer outputs probabilities for each digit (0–9).
        - The highest probability = the predicted digit.
        """)
    else:
        st.warning("⚠️ Please draw a digit before clicking Predict!")

# ------------------------------
# 4️⃣ Sidebar Information
# ------------------------------
st.sidebar.title("📘 About this App")
st.sidebar.markdown("""
**Deep Learning Concept Recap**
- **Model Persistence:** We load the pre-trained CNN (`mnist_cnn_model.h5`) instead of retraining.
- **Input Preprocessing:** Resized and normalized user drawing (28×28, grayscale, /255).
- **Inference:** Model predicts probabilities via Softmax.
- **Dropout (optional):** Prevents overfitting by randomly deactivating neurons.

**Real-world Examples**
- 🏦 Bank check digit readers  
- 🏫 Smart classroom attendance  
- 📬 Postal code recognition  
""")

st.sidebar.info("Developed as part of **DL(easy) – Handwriting Detection Project (Day 3)**")

# ------------------------------
# 5️⃣ Run Locally
# ------------------------------
# To run:
# streamlit run streamlit_app.py
