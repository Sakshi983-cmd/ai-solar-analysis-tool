import streamlit as st
from PIL import Image
import numpy as np
import cv2
from transformers import pipeline

# Set up the app
st.set_page_config(page_title="Solar Rooftop AI Report", layout="centered")
st.title("☀️ Solar Rooftop Analysis Tool")

@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="gpt2", device=-1)

generator = load_generator()

uploaded_file = st.file_uploader("Upload Rooftop Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    resized_img = img.resize((400, 400))
    st.image(resized_img, caption="Uploaded Image", use_column_width=True)

    # Edge detection
    img_np = np.array(resized_img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    coverage = np.sum(edges > 0) / (400 * 400) * 100

    st.write(f"📐 Rooftop Coverage (Edge Detection): **{coverage:.2f}%**")

    st.write("🧠 Generating Solar Feasibility Report...")
    prompt = (
        f"The rooftop image shows {coverage:.2f}% usable area. Generate a solar installation feasibility report "
        f"including capacity, cost, and benefits."
    )
    
    result = generator(prompt, max_new_tokens=150, truncation=True, pad_token_id=50256)[0]['generated_text']
    
    st.subheader("📋 AI-Powered Report")
    st.write(result)
