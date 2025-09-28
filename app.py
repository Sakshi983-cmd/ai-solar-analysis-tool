import streamlit as st
from PIL import Image
from transformers import pipeline

# Load a lightweight LLM (distilgpt2) for stability
try:
    generator = pipeline("text-generation", model="distilgpt2")
except Exception as e:
    generator = None

# Placeholder for rooftop area estimation
def estimate_rooftop_area(image):
    return "Approx. 120 sq meters"

# Generate LLM-based solar analysis
def get_llm_analysis(area="120 sq meters"):
    if generator is None:
        return "LLM model could not be loaded. Please try again later."
    
    prompt = f"""
    Suggest a solar installation plan for a rooftop of {area}.
    Include:
    - Estimated solar potential (kWh/year)
    - Panel recommendation
    - Installation steps
    - Maintenance tips
    - Cost & ROI
    - Regulatory compliance
    - Confidence score
    """
    try:
        response = generator(prompt, max_length=200, do_sample=True, temperature=0.7)[0]['generated_text']
    except Exception as e:
        response = f"LLM generation failed: {e}"
    return response

# Streamlit UI
def main():
    st.title("☀️ Solar Industry AI Assistant (LLM-Powered)")
    st.write("Upload a rooftop satellite image to get AI-generated solar installation recommendations.")

    uploaded_file = st.file_uploader("Upload rooftop satellite image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).resize((512, 512))
        st.image(image, caption="Uploaded Rooftop Image", use_column_width=True)

        if st.button("Analyze Rooftop"):
            st.info("Analyzing rooftop using LLM...")
            area = estimate_rooftop_area(image)
            st.subheader("Estimated Rooftop Area")
            st.write(area)

            result_text = get_llm_analysis(area)
            st.subheader("LLM-Based Solar Recommendation")
            st.write(result_text)

if __name__ == "__main__":
    main()
