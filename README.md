🩺 MedScan AI

AI-powered Chest X-Ray Disease Detection & Automated Medical Report Generation

MedScan AI is an advanced deep-learning application that analyzes chest X-ray images and predicts possible lung diseases using a fine-tuned VGG16 model. The system also generates a structured medical-style report using the Google Gemini API, making it useful for hospitals, radiologists, and health-tech applications.

📌 Key Features
✅ 1. Deep Learning–based Disease Detection

Fine-tuned VGG16 architecture

Data augmentation: rotation, zoom, flip

Handles class imbalance using class weights

Achieves high accuracy on custom dataset

✅ 2. Automated Report Generation

Google Gemini API generates:

Disease summary

Medical-style interpretation

Risk category

Possible causes

Recommended next steps

✅ 3. Simple & Interactive UI

Built using Streamlit

Upload X-ray

View prediction & probability

Download complete report

🧠 How It Works (Pipeline)
User Uploads X-Ray → Preprocessing → VGG16 Prediction → Gemini API → Report → Download as PDF

🏗️ Tech Stack

Deep Learning: TensorFlow, Keras (VGG16)
Computer Vision: OpenCV, NumPy
UI: Streamlit
Report Generation: Google Gemini API
Visualization: Matplotlib

📂 Project Structure
MedScanAI/
│── app.py                        # Streamlit UI
│── model/
│   ├── vgg16_model.h5           # Trained model
│── utils/
│   ├── preprocess.py            # Image preprocessing
│   ├── report_generator.py      # Gemini API integration
│── data/
│── README.md


🧪 Model Details

Base Model: VGG16 pretrained on ImageNet

Fine-tuning:

Unfrozen last few convolution blocks

Dense layers added for classification

Augmentation:

RandomRotation

RandomZoom

HorizontalFlip

VerticalFlip

Optimization:

Adam optimizer

EarlyStopping

ReduceLROnPlateau

📝 Sample Report Output

Your report includes:

✔ Disease prediction

✔ Confidence score

✔ Critical/non-critical classification

✔ Radiology-style explanation

✔ Suggested next steps

📸 Screenshots

(Add UI screenshots here later if you want.)

📌 Future Improvements

Add multiple disease detection

Implement Grad-CAM heatmaps

Deploy on cloud

Add patient history support

👤 Author

Mrigank Mathur
AI/ML Engineer | Deep Learning Enthusiast
