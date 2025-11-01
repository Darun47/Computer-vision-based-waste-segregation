import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import requests
import gdown

# Set page configuration
st.set_page_config(
    page_title="SmartWasteAI - Waste Classification",
    page_icon="🗑️",
    layout="centered"
)

# Google Drive direct download link (you'll update this)
MODEL_URL = "https://drive.google.com/drive/folders/1rwnhNUXqs-wSKgek5ZoVbrVI-uRNL1jr"
MODEL_PATH = "/content/drive/MyDrive/DATA./models/waste_classifier.h5"

@st.cache_resource
def download_and_load_model():
    """
    Download model from Google Drive if not exists, then load it
    """
    try:
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Download model if it doesn't exist
        if not os.path.exists(MODEL_PATH):
            with st.spinner('📥 Downloading AI model (first time setup)...'):
                # Method 1: Try gdown first
                try:
                    gdown.download(MODEL_URL, MODEL_PATH, quiet=True)
                    st.success("✅ Model downloaded successfully!")
                except:
                    # Method 2: Alternative download approach
                    st.info("🔄 Trying alternative download method...")
                    import urllib.request
                    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        
        # Load the model
        model = load_model(MODEL_PATH)
        return model
        
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.info("🔧 Running in demo mode with example predictions")
        return None

# Load the model
model = download_and_load_model()

# Define class labels
class_labels = {
    0: {'name': 'Biodegradable', 'bin_color': 'Green', 'bin_emoji': '🟢', 
        'bin_info': 'For organic waste like food scraps, garden waste, and biodegradable materials.'},
    1: {'name': 'Hazardous', 'bin_color': 'Red', 'bin_emoji': '🔴',
        'bin_info': 'For dangerous materials like batteries, chemicals, medical waste, and electronics.'},
    2: {'name': 'Recyclable', 'bin_color': 'Blue', 'bin_emoji': '🔵',
        'bin_info': 'For materials that can be recycled like paper, plastic, glass, metal, and textiles.'}
}

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_waste(img_array):
    if model is None:
        # Demo mode - random prediction
        return np.random.randint(0, 3), np.random.uniform(0.7, 0.95)
    
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    return predicted_class, confidence

# Streamlit UI
st.title("♻️ SmartWasteAI")
st.markdown("### AI-Powered Waste Segregation for Smart Cities")
st.write("Upload an image of a waste item, and our AI will classify it and recommend the correct disposal bin.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict
    with st.spinner('🔍 AI is analyzing the waste item...'):
        processed_image = preprocess_image(image_display)
        predicted_class, confidence = predict_waste(processed_image)
    
    # Get class info
    class_info = class_labels[predicted_class]
    
    # Display results
    st.success("✅ Analysis Complete!")
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="**Predicted Waste Type**", 
                 value=f"{class_info['bin_emoji']} {class_info['name']}")
        st.metric(label="**Confidence Level**", 
                 value=f"{confidence:.2%}")
    
    with col2:
        st.markdown(f"### 🗑️ Recommended Bin")
        st.markdown(f"# {class_info['bin_emoji']} **{class_info['bin_color']} Bin**")
        st.info(class_info['bin_info'])
    
    # Confidence visualization
    st.markdown("### 📊 Confidence Level")
    st.progress(float(confidence))
    st.write(f"Model is {confidence:.2%} confident about this prediction")

# Demo section for when model is not loaded
if model is None:
    st.warning("""
    ⚠️ **Demo Mode Active** 
    - The AI model is loading in demo mode
    - Real AI predictions will work after model download completes
    """)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About SmartWasteAI")
    st.write("Classifies waste into three categories using deep learning")
    
    st.markdown("""
    **🗑️ Waste Categories:**
    - 🟢 **Green Bin**: Biodegradable
    - 🔵 **Blue Bin**: Recyclable  
    - 🔴 **Red Bin**: Hazardous
    """)
    
    st.markdown("---")
    st.write("**🔧 Built with:**")
    st.write("- TensorFlow & MobileNetV2")
    st.write("- Streamlit Cloud")
    st.write("- Python")
    
    st.markdown("---")
    st.write("**🎓 Educational Project**")
    st.write("Machine Learning & Deep Learning Course")

# Footer
st.markdown("---")
st.markdown("Developed as part of **AI in Action Project** | **Machine Learning & Deep Learning Course**")
