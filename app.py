import streamlit as st
from PIL import Image

def get_dummy_analysis():
    # Ye dummy data aapke AI response jaisa lagega
    return {
        "solar_potential": "Estimated 4500 kWh/year",
        "panel_recommendation": "20 pcs of 320W Monocrystalline panels",
        "installation_steps": "Roof inspection, mounting, electrical wiring, permits",
        "maintenance_tips": "Quarterly cleaning, yearly inspection, monitor via app",
        "cost_roi": "Approx cost $12,000, payback period 6 years, ROI 15%",
        "regulations": "Compliant with local net metering rules and safety standards",
        "confidence_score": "High (92%)"
    }

def main():
    st.title("Solar Industry AI Assistant (Demo without API key)")
    st.write("Upload rooftop satellite image and get dummy solar installation assessment.")

    uploaded_file = st.file_uploader("Upload rooftop satellite image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Analyze Rooftop"):
            st.info("Showing dummy analysis (No API used)")
            result = get_dummy_analysis()
            st.subheader("Solar Potential")
            st.write(result["solar_potential"])

            st.subheader("Panel Recommendation")
            st.write(result["panel_recommendation"])

            st.subheader("Installation Steps")
            st.write(result["installation_steps"])

            st.subheader("Maintenance Tips")
            st.write(result["maintenance_tips"])

            st.subheader("Cost & ROI")
            st.write(result["cost_roi"])

            st.subheader("Regulations")
            st.write(result["regulations"])

            st.subheader("Confidence Score")
            st.write(result["confidence_score"])

if __name__ == "__main__":
    main()
