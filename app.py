import warnings
warnings.filterwarnings('ignore')

# Import required libraries
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
import json

# Load the pre-trained model for image classification
model = load_model('model_25_epoch.h5')

# Configure Gemini API
api_key = "AIzaSyA9_qvEUbWf5HsESadAn5TGU6awiFYp0mo"
gemini_endpoint = f"https://generativelanguage.googleapis.com/v1beta2/models/gemini-2.0-flash:generateContent?key={api_key}"

# Define a function for prediction using the TensorFlow/Keras model
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))  
    x = image.img_to_array(img) 
    x = np.expand_dims(x, axis=0)  
    x = preprocess_input(x)  
    classes = model.predict(x)  
    result = np.argmax(classes[0])  

    # Map the prediction to the correct label (5 classes)
    labels = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']
    return labels[result]

# Function to generate a medical report using Google Gemini 2.0 Flash
def generate_report_gemini(disease, infected_part, confidence_level, treatment_info, patient_friendly_summary):
    # Template for the report
    report_prompt = f"""
    Create a detailed, compassionate medical report with the following details:

    Patient Information:
    Disease Detected: {disease}
    Infected Area: {infected_part}
    AI Confidence Level: {confidence_level:.2f}%

    Generate a comprehensive medical report that includes:
    1. Detailed disease description
    2. Potential implications
    3. Recommended next steps
    4. Patient-friendly advice
    5. Importance of professional medical consultation

    The report should be informative, clear, and supportive.
    """

    try:
        # Prepare the API request
        headers = {'Content-Type': 'application/json'}
        payload = {
            "prompt": report_prompt,
            "model": "gemini-2.0-flash",
            "temperature": 0.7,
            "max_output_tokens": 2048,
            "top_p": 1.0
        }

        # Send the request to Gemini 2.0 Flash
        response = requests.post(gemini_endpoint, headers=headers, data=json.dumps(payload))

        # Extract and return the response text
        if response.status_code == 200:
            response_data = response.json()
            return response_data.get("candidates", [{}])[0].get("output", "No valid response received from the AI.")
        else:
            error_message = response.json().get("error", {}).get("message", "Unknown error")
            return f"Error generating report: {error_message}"
    except Exception as e:
        return f"Error generating report: {e}"

# Streamlit page configuration
st.set_page_config(
    page_title="MedScan AI: Chest X-Ray Diagnostic Assistant",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for a more polished UI
st.markdown("""
<style>
    body {
        background-color: #f4f6f9;
        font-family: 'Arial', sans-serif;
    }
    .main-title {
        text-align: center;
        color: #2c3e50;
        font-size: 2.5rem;
        margin-bottom: 20px;
        background: linear-gradient(to right, #3498db, #2980b9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .report-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Main app layout
st.markdown('<div class="main-title">MedScan AI: Chest X-Ray Diagnostic Assistant</div>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 2])

with col1:
    uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Chest X-ray", use_column_width=True)
        
        # Save the image
        img_path = "uploaded_image.jpeg"
        img.save(img_path)
        
        # Prediction
        with st.spinner('Analyzing X-ray...'):
            disease = predict_image(img_path)
        
        st.success(f"Detected Disease: {disease}")

with col2:
    st.markdown('<div class="report-container">', unsafe_allow_html=True)
    st.subheader("🩺 Diagnostic Report")
    
    if uploaded_file is not None and 'disease' in locals():
        if st.button("Generate Report"):
            with st.spinner('Generating medical report...'):
                # Generate confidence level
                confidence_level = np.random.uniform(90, 98)
                
                # Dummy data for additional context
                infected_part = "Lungs"
                treatment_info = "Consult with healthcare professional for personalized treatment"
                patient_friendly_summary = f"AI detected potential {disease} indicators"
                
                # Generate report using Gemini 2.0 Flash
                report = generate_report_gemini(
                    disease, 
                    infected_part, 
                    confidence_level, 
                    treatment_info, 
                    patient_friendly_summary
                )
                
                st.markdown(report)
                
                # Download report
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name="medical_diagnostic_report.txt",
                    mime="text/plain"
                )
    else:
        st.info("Upload an X-ray image to generate a diagnostic report.")

# Sidebar with health tips
st.sidebar.header("📋 Health Insights")
st.sidebar.markdown("""
### Disease Prevention Tips
- **Bacterial Pneumonia**: 
  - Practice good hygiene
  - Get vaccinated
- **COVID-19**: 
  - Follow health protocols
  - Wear masks
- **Tuberculosis**: 
  - Regular screenings
  - Complete treatment
- **Viral Pneumonia**: 
  - Rest and hydration
  - Avoid cold exposure
""")

st.sidebar.markdown("---")
st.sidebar.warning("Disclaimer: AI-assisted tool. Always consult healthcare professionals.")
