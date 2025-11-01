import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

# Set page config
st.set_page_config(
    page_title="Waste Segregation AI",
    page_icon="🗑️",
    layout="centered"
)

# Title and description
st.title("🗑️ AI Waste Segregation System")
st.markdown("Upload an image of waste item to classify it as Biodegradable, Hazardous, or Recyclable")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a waste image...", 
    type=['jpg', 'jpeg', 'png']
)

# Load your model (adjust this based on your actual model)
@st.cache_resource
def load_model():
    # Replace this with your actual model loading code
    # Example:
    # model = torch.load('waste_model.pth')
    # model.eval()
    
    # For now, returning None as placeholder
    return None

model = load_model()

# Class names (adjust based on your model)
class_names = ['Biodegradable', 'Hazardous', 'Recyclable']

def preprocess_image(image):
    """Preprocess the image for model prediction"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict_waste(image, model):
    """Make prediction on the image"""
    # Preprocess
    processed_image = preprocess_image(image)
    
    # Make prediction (replace with your actual prediction code)
    # with torch.no_grad():
    #     outputs = model(processed_image)
    #     probabilities = torch.nn.functional.softmax(outputs, dim=1)
    #     predicted_class = torch.argmax(probabilities, 1)
    
    # For demo purposes - replace with your actual model prediction
    # This is just placeholder logic
    probabilities = [0.00, 1.00, 0.00]  # Example: Hazardous
    predicted_idx = 1  # Hazardous
    
    return predicted_idx, probabilities

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Make prediction
    with st.spinner('Analyzing waste type...'):
        predicted_idx, probabilities = predict_waste(image, model)
    
    # Display results
    st.subheader("🔍 Prediction Results")
    
    # Color coding for bins
    bin_colors = {
        'Biodegradable': '🟢 Green Bin',
        'Hazardous': '🔴 Red Bin', 
        'Recyclable': '🔵 Blue Bin'
    }
    
    predicted_class = class_names[predicted_idx]
    recommended_bin = bin_colors[predicted_class]
    
    # Prediction and confidence
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Prediction",
            value=predicted_class,
            delta="High Confidence" if max(probabilities) > 0.8 else "Medium Confidence"
        )
    
    with col2:
        st.metric(
            label="Confidence",
            value=f"{max(probabilities)*100:.2f}%"
        )
    
    # Recommended bin
    st.info(f"**Recommended Bin:** {recommended_bin}")
    
    # Probabilities chart
    st.subheader("📊 All Probabilities")
    
    for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
        percentage = prob * 100
        
        # Progress bar with color coding
        if class_name == predicted_class:
            st.progress(prob, text=f"**{class_name}: {percentage:.2f}%**")
        else:
            st.progress(prob, text=f"{class_name}: {percentage:.2f}%")

# Add some information
st.markdown("---")
st.markdown("""
### 🎯 How to Use:
1. Upload a clear image of a waste item
2. Wait for AI analysis
3. Check the prediction and recommended disposal bin

### 📁 Supported Waste Types:
- **Biodegradable**: Food waste, paper, garden waste
- **Hazardous**: Batteries, chemicals, electronics  
- **Recyclable**: Plastic, glass, metal, cardboard
""")
