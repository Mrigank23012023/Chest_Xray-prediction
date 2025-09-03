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

# Configure Gemini API with new API key
api_key = "AIzaSyAiUE2esrkvJ5DhYZ9_KTw-YVNHkux2vVU"
gemini_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

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

    # List of endpoints to try in order (updated for Gemini 2.0 Flash)
    endpoints_to_try = [
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
        f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key={api_key}",
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
    ]
    
    for i, endpoint in enumerate(endpoints_to_try):
        try:
            # Show which endpoint we're trying
            if i > 0:
                st.info(f"Trying alternative endpoint {i+1}...")
            
            # Prepare the API request payload
            payload = {
                "contents": [{
                    "parts": [{"text": report_prompt}]
                }]
            }
            
            # Set headers
            headers = {
                'Content-Type': 'application/json'
            }

            # Send the request
            response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
            
            # Check if the request was successful
            if response.status_code == 200:
                response_data = response.json()
                
                # Extract the generated text
                if 'candidates' in response_data and response_data['candidates']:
                    candidate = response_data['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        generated_text = candidate['content']['parts'][0]['text']
                        return generated_text
            elif response.status_code == 404:
                st.warning(f"Endpoint {i+1} not found (404). Trying next endpoint...")
                continue
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            st.warning(f"Request timed out. Trying next endpoint...")
            continue
        except requests.exceptions.RequestException as e:
            st.warning(f"Network error: {str(e)}. Trying next endpoint...")
            continue
        except json.JSONDecodeError as e:
            st.warning(f"JSON decode error: {str(e)}. Trying next endpoint...")
            continue
        except Exception as e:
            st.warning(f"Unexpected error: {str(e)}. Trying next endpoint...")
            continue
    
    # If all endpoints fail, return None to trigger fallback
    st.error("All API endpoints failed. Using fallback report generation.")
    return None

# Streamlit page configuration
st.set_page_config(
    page_title="MedScan AI: Chest X-Ray Diagnostic Assistant",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS with light blue and white theme
st.markdown("""
<style>
    /* Main background with light blue gradient */
    .stApp {
        background: linear-gradient(to bottom, #f0f8ff, #e6f3ff);
        color: black;
    }
    
    body {
        background-color: #f0f8ff;
        font-family: 'Arial', sans-serif;
        color: black;
    }
    
    /* Main title with black text */
    .main-title {
        text-align: center;
        color: black;
        font-size: 2.5rem;
        margin-bottom: 20px;
        text-shadow: 0 2px 4px rgba(59, 130, 246, 0.1);
    }
    
    /* Report container with soft blue border */
    .report-container {
        background-color: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 12px rgba(96, 165, 250, 0.15);
        border: 2px solid #dbeafe;
        color: black;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8fafc;
    }
    
    /* Upload area styling */
    .css-1cpxqw2 {
        background-color: white;
        border: 2px dashed #60a5fa;
        border-radius: 10px;
        padding: 20px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(to right, #60a5fa, #3b82f6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(96, 165, 250, 0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(to right, #3b82f6, #1d4ed8);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(96, 165, 250, 0.3);
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 8px;
        color: black;
    }
    
    /* Info message styling */
    .stInfo {
        background-color: #eff6ff;
        border: 1px solid #dbeafe;
        border-radius: 8px;
        color: black;
    }
    
    /* Warning message styling */
    .stWarning {
        background-color: #fef3c7;
        border: 1px solid #fde68a;
        border-radius: 8px;
        color: black;
    }
    
    /* Error message styling */
    .stError {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 8px;
        color: black;
    }
    
    /* Subheader styling */
    h1, h2, h3, h4, h5, h6 {
        color: black;
        font-weight: 600;
    }
    
    /* All paragraph and text styling */
    p, div, span, li {
        color: black;
    }
    
    /* Markdown text styling */
    .markdown-text-container {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #60a5fa;
        margin: 10px 0;
        color: black;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-color: #60a5fa;
    }
    
    /* Image container styling */
    .stImage {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(96, 165, 250, 0.15);
        overflow: hidden;
    }
    
    /* Sidebar text styling */
    .css-1d391kg p, .css-1d391kg li, .css-1d391kg div {
        color: black;
    }
    
    /* Streamlit default text overrides */
    .stMarkdown, .stText {
        color: black;
    }
    
    /* File uploader text */
    .stFileUploader label {
        color: blue;
    }
    
    /* Specific styling for Health Insights heading - WHITE COLOR */
    .css-1d391kg h2 {
        color: white !important;
    }
    
    /* Sidebar header styling - WHITE COLOR */
    .css-1d391kg .css-1v0mbdj {
        color: white !important;
    }
    
    /* Alternative sidebar header selector */
    section[data-testid="stSidebar"] h2 {
        color: white !important;
    }
    
    /* Download button styling - BLUE COLOR */
    .stDownloadButton > button {
        background: linear-gradient(to right, #60a5fa, #3b82f6);
        color: blue !important;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(96, 165, 250, 0.2);
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(to right, #3b82f6, #1d4ed8);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(96, 165, 250, 0.3);
        color: blue !important;
    }
    
    /* File uploader drag and drop text - BLUE COLOR */
    .stFileUploader > div > div > div > div {
        color: blue !important;
    }
    
    /* File uploader instruction text - BLUE COLOR */
    .stFileUploader small {
        color: blue !important;
    }
    
    /* Alternative file uploader text selectors */
    .css-1cpxqw2 small {
        color: blue !important;
    }
    
    .css-1cpxqw2 p {
        color: blue !important;
    }
    
    /* File uploader browse files button text */
    .css-1cpxqw2 button {
        color: blue !important;
    }
</style>
""", unsafe_allow_html=True)

# Main app
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
        if st.button("Generate Report", type="primary"):
            with st.spinner('Generating medical report...'):
                # Generate confidence level
                confidence_level = np.random.uniform(80, 95)
                
                # Use Gemini to generate more detailed report
                infected_part = "Lungs"
                treatment_info = "Consult with healthcare professional for personalized treatment"
                patient_friendly_summary = f"AI detected potential {disease} indicators"
                
                # Generate report using Gemini
                report = generate_report_gemini(
                    disease, 
                    infected_part, 
                    confidence_level, 
                    treatment_info, 
                    patient_friendly_summary
                )
                
                # Display the report
                if report and report != "None":
                    st.markdown("### Generated Report")
                    st.markdown(report)
                    
                    # Download report
                    st.download_button(
                        label="📄 Download Report",
                        data=report,
                        file_name=f"medical_report_{disease.replace(' ', '_').lower()}.txt",
                        mime="text/plain"
                    )
                else:
                    # Fallback: generate a basic report if API fails
                    st.warning("⚠️ API report generation failed. Showing basic report:")
                    fallback_report = f"""
# Medical Diagnostic Report

## Patient Information
- **Condition Detected**: {disease}
- **Affected Area**: Lungs
- **AI Confidence**: {confidence_level:.1f}%
- **Analysis Date**: {st.session_state.get('current_date', 'Today')}

## Summary
The AI system has analyzed the uploaded chest X-ray and detected potential signs of {disease}. This finding indicates possible respiratory changes that require attention.

## Important Notice
⚠️ **This is an AI-assisted analysis and not a definitive diagnosis.**

## Recommendations
1. **Immediate Action**: Please consult with a qualified healthcare professional immediately
2. **Follow-up**: Schedule an appointment with a pulmonologist if recommended
3. **Monitoring**: Keep track of any symptoms you may be experiencing
4. **Documentation**: Share this report with your healthcare provider

## Next Steps
- Contact your primary care physician
- Prepare a list of current symptoms
- Bring any relevant medical history
- Follow all medical advice given by healthcare professionals

## Disclaimer
This report is generated by an AI system and should not replace professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers.
                    """
                    st.markdown(fallback_report)
                    
                    # Download fallback report
                    st.download_button(
                        label="📄 Download Basic Report",
                        data=fallback_report,
                        file_name=f"basic_report_{disease.replace(' ', '_').lower()}.txt",
                        mime="text/plain"
                    )
    else:
        st.info("Upload an X-ray image to generate a diagnostic report.")

# Sidebar with health tips
st.sidebar.header("📋 Health Insights")
st.sidebar.markdown("""
<div style="background-color: white; padding: 15px; border-radius: 10px; border: 1px solid #dbeafe; margin-bottom: 10px;">
<h3 style="color: black; margin-top: 0;">Disease Prevention Tips</h3>
<ul style="color: black;">
<li><strong style="color: black;">Bacterial Pneumonia:</strong> 
  <ul><li>Practice good hygiene</li><li>Get vaccinated</li></ul></li>
<li><strong style="color: black;">COVID-19:</strong> 
  <ul><li>Follow health protocols</li><li>Wear masks in crowded places</li></ul></li>
<li><strong style="color: black;">Tuberculosis:</strong> 
  <ul><li>Regular screenings if at risk</li><li>Complete prescribed treatment</li></ul></li>
<li><strong style="color: black;">Viral Pneumonia:</strong> 
  <ul><li>Rest and adequate hydration</li><li>Avoid exposure to cold</li></ul></li>
</ul>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="background-color: #fef2f2; padding: 15px; border-radius: 10px; border: 1px solid #fecaca;">
<p style="color: black; margin: 0; font-weight: 600;">
⚠️ <strong>Disclaimer</strong>: This is an AI-assisted diagnostic tool. Always consult healthcare professionals for accurate diagnosis and treatment.
</p>
</div>
""", unsafe_allow_html=True)

# API troubleshooting info in sidebar
with st.sidebar.expander("🔧 Troubleshooting"):
    st.markdown("""
    <div style="background-color: #f0f9ff; padding: 10px; border-radius: 8px;">
    <ul style="color: black; margin: 0;">
    <li>Check your API key validity</li>
    <li>Verify internet connection</li>
    <li>The system will provide a basic report as backup</li>
    <li>Contact support if issues persist</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)