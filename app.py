import streamlit as st
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
import io

# LLM model 
solar_llm = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

# Title
st.title("☀️ Solar Rooftop Potential Analyzer - AI Assistant")

# Upload section
uploaded_file = st.file_uploader("Upload a rooftop image (satellite or aerial view)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_container_width=True)

    # Convert image to grayscale
    gray_img = np.array(image.convert("L"))
    st.image(gray_img, caption="Grayscale Image", use_container_width=True)

    # Edge detection
    edges = cv2.Canny(gray_img, 100, 200)
    st.image(edges, caption="Edge Detection", use_container_width=True)

    # Estimate rooftop area (based on white pixel count)
    rooftop_area = np.sum(edges > 0)
    height, width = edges.shape
    total_area = height * width
    area_ratio = rooftop_area / total_area
    solar_score = round(area_ratio * 10, 2)

    st.subheader("📐 Image Analysis")
    st.write(f"Image Dimensions: {width} x {height} pixels")
    st.write(f"Estimated Rooftop Area (pixels): {rooftop_area}")
    st.write(f"Solar Potential Score: **{solar_score} / 10**")

    # Generate LLM report
    prompt = (
        f"This rooftop image has a detected usable area of {rooftop_area} pixels and a solar score of {solar_score}/10. "
        "Based on this, provide suggestions for solar panel installation and estimate monthly savings, payback period, and total ROI over 25 years."
    )

    with st.spinner("Generating AI report..."):
        ai_report = solar_llm(prompt, max_length=250, do_sample=True, temperature=0.7)[0]['generated_text']

    st.subheader("🧠 AI Assistant Report")
    st.markdown(ai_report)

    st.markdown("---")
    st.caption("🔒 No OpenAI key used — runs fully on open-source LLM (GPT-Neo)")
