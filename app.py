import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
from PIL import Image
import cv2
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# -------------------- Page Setup --------------------
st.set_page_config(page_title="AI Solar Analysis", layout="centered")

st.title("☀️ AI-Powered Solar Rooftop Analysis")
st.write("Upload an image of a rooftop to receive AI-generated solar feasibility insights.")

# -------------------- Model Setup --------------------
@st.cache_resource
def load_model():
    model_name = "EleutherAI/gpt-neo-1.3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# -------------------- Image Upload --------------------
uploaded_file = st.file_uploader("📷 Upload Rooftop Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((400, 400))
    st.image(image, caption="Uploaded Rooftop", use_column_width=True)

    # -------------------- Image Processing --------------------
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    rooftop_area = np.sum(edges > 0) / (400 * 400) * 100  # %
    st.write(f"📐 Estimated rooftop edge coverage: **{rooftop_area:.2f}%**")

    # -------------------- LLM Prompt --------------------
    prompt = (
        f"A 400x400 rooftop image shows {rooftop_area:.2f}% rooftop edge coverage. "
        "The rooftop is flat with minimal obstruction. "
        "Estimate solar panel capacity (kW), approximate energy generation per year (kWh), "
        "expected cost, ROI, and payback period for installing solar panels."
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=250)
    st.write("🧠 Generating AI analysis...")

    from transformers import TextGenerationPipeline
    gen_pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=-1)

    output = gen_pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)[0]['generated_text']
    
    # -------------------- Result --------------------
    st.subheader("📋 AI Report")
    st.write(output)
