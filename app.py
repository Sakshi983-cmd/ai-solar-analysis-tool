import streamlit as st
from PIL import Image
from transformers import pipeline

# Load GPT-Neo model (open-source, no API key needed)
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

# Placeholder for rooftop area estimation (Vision AI can be added later)
def estimate_rooftop_area(image):
    # Future: Use segmentation model here
    return "Approx. 120 sq meters"

# Generate LLM-based solar analysis
def get_llm_analysis(area="120 sq meters"):
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
    response = generator(prompt, max_length=300, do_sample=True, temperature=0.7)[0]['generated_text']
    return response

# Streamlit UI
def main():
    st.title("☀️ Solar Industry AI Assistant (GPT-Neo Powered)")
    st.write("Upload a rooftop satellite image to get AI-generated solar installation recommendations.")

    uploaded_file = st.file_uploader("Upload rooftop satellite image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Rooftop Image", use_column_width=True)

        if st.button("Analyze Rooftop"):
            st.info("Analyzing rooftop using GPT-Neo...")
            area = estimate_rooftop_area(image)
            st.subheader("Estimated Rooftop Area")
            st.write(area)

            result_text = get_llm_analysis(area)
            st.subheader("LLM-Based Solar Recommendation")
            st.write(result_text)

if __name__ == "__main__":
    main()
