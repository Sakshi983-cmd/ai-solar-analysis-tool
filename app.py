import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
from llama_cpp import Llama  # Local LLM integration

# Initialize LLaMA model (update path as per your downloaded model)
llm = Llama(model_path="models/llama-7b.ggmlv3.q4_0.bin")

st.set_page_config(page_title="Solar Rooftop Analysis with Local LLM", layout="wide")
st.title("☀️ AI-Powered Rooftop Solar Analysis Tool with Local LLM")

st.markdown("""
Upload a rooftop image (satellite or drone view) to get a simulated solar potential assessment.
Ask questions about solar panels and installation — answers powered by a local LLM, no API key required.
""")

uploaded_file = st.file_uploader("Upload a rooftop image (JPG/PNG)", type=["jpg", "jpeg", "png"])

def analyze_image(pil_img):
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    def to_bytes(image, cmap=None):
        buf = BytesIO()
        plt.figure(figsize=(5, 3))
        if cmap:
            plt.imshow(image, cmap=cmap)
        else:
            plt.imshow(image)
        plt.axis("off")
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf

    width, height = pil_img.size
    rooftop_area = (0.8 - 0.2) * (0.7 - 0.4)
    simulated_area = width * height * rooftop_area

    report = f"""
### Simulated Solar Potential Assessment

**Image Dimensions:** {width} x {height} pixels  
**Estimated Rooftop Area (pixels):** {simulated_area:.2f}  

---

**Solar Potential Score:** 6.5 / 10  
**Installation Suggestion:**  
- Estimated fit: 10–12 standard panels  
- Avoid shadowed or obstructed areas  
- Place panels on flat, unshaded zones  

**ROI Estimate (25 years):**  
- **Monthly Savings:** $60–90  
- **Payback Period:** 9–13 years  
- **Lifetime Savings:** $18,000 – $27,000

---

_Note: Simulated data. Accuracy depends on precise shadow/angle/sunlight mapping._
"""
    return to_bytes(pil_img), to_bytes(gray, cmap='gray'), to_bytes(edges, cmap='gray'), report

# Function to query the local LLM
def ask_local_llm(prompt):
    response = llm(prompt, max_tokens=150)
    return response['choices'][0]['text'].strip()

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.success("Image uploaded successfully. Processing...")

    img_orig, img_gray, img_edges, result = analyze_image(pil_img)

    st.subheader("Rooftop Image Analysis")
    c1, c2, c3 = st.columns(3)
    c1.image(img_orig, caption="Original", use_column_width=True)
    c2.image(img_gray, caption="Grayscale", use_column_width=True)
    c3.image(img_edges, caption="Edge Detection", use_column_width=True)

    st.subheader("Solar Potential Analysis")
    st.markdown(result)

    st.subheader("Ask about Solar Panels & Installation")
    user_question = st.text_input("Type your question here:")
    if user_question:
        with st.spinner("Generating answer..."):
            answer = ask_local_llm(user_question)
        st.markdown(f"**Answer:** {answer}")

    st.markdown("Developed for Solar Industry AI Assistant Internship ✅")
else:
    st.info("Please upload a rooftop image to begin.")
