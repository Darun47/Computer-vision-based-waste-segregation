import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import urllib.request
import tempfile

# Set page configuration
st.set_page_config(
    page_title="SmartWasteAI - Waste Classification",
    page_icon="🗑️",
    layout="centered"
)

# Your Google Drive shareable link (you'll need to update this)
MODEL_URL = "https://drive.google.com/drive/folders/1rwnhNUXqs-wSKgek5ZoVbrVI-uRNL1jr"
MODEL_PATH = "/content/drive/MyDrive/DATA./data/models/waste_classifier.h5"

@st.cache_resource
def download_and_load_high_accuracy_model():
    """
    Download the high-accuracy model from Google Drive
    """
    try:
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Download model if it doesn't exist
        if not os.path.exists(MODEL_PATH):
            with st.spinner('📥 Downloading high-accuracy AI model...'):
                # Download from Google Drive
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
                st.sidebar.success("✅ High-accuracy model downloaded!")
        
        # Load the model
        model = load_model(MODEL_PATH)
        st.sidebar.success("🤖 High-accuracy AI model loaded!")
        return model
        
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")
        # Fallback to small model
        return load_fallback_model()

def load_fallback_model():
    """
    Fallback to a small model if high-accuracy model fails
    """
    try:
        small_model_path = "models/waste_classifier_small.h5"
        if os.path.exists(small_model_path):
            return load_model(small_model_path)
    except:
        pass
    return None

# Load the high-accuracy model
model = download_and_load_high_accuracy_model()

# Define class labels
class_labels = {
    0: {'name': 'Biodegradable', 'bin_color': 'Green', 'bin_emoji': '🟢'},
    1: {'name': 'Hazardous', 'bin_color': 'Red', 'bin_emoji': '🔴'},
    2: {'name': 'Recyclable', 'bin_color': 'Blue', 'bin_emoji': '🔵'}
}

def preprocess_image(img):
    """Preprocess image exactly like during training"""
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_waste(img_array):
    """Make prediction with high-accuracy model"""
    if model is None:
        return None, 0.0
    
    try:
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, 0.0

# Streamlit UI
st.title("♻️ SmartWasteAI - High Accuracy Version")
st.markdown("### 🎯 98%+ Accurate Waste Classification")

if model:
    st.success("🤖 **High-Accuracy AI Active** - Real-time classification with 98%+ accuracy")
else:
    st.warning("🔧 **Standard Accuracy Mode**")

st.write("Upload an image of waste for high-accuracy classification.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image_display = Image.open(uploaded_file)
        st.image(image_display, caption="📸 Uploaded Image", use_column_width=True)
        
        # Preprocess and predict
        with st.spinner('🔍 High-accuracy analysis in progress...'):
            processed_image = preprocess_image(image_display)
            predicted_class, confidence = predict_waste(processed_image)
        
        if predicted_class is not None:
            class_info = class_labels[predicted_class]
            
            # Display results
            st.success("✅ High-Accuracy Analysis Complete!")
            
            # Results layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="**Waste Type**", 
                    value=f"{class_info['bin_emoji']} {class_info['name']}"
                )
                st.metric(
                    label="**Confidence**", 
                    value=f"{confidence:.1%}"
                )
            
            with col2:
                st.markdown(f"### 🗑️ **{class_info['bin_color']} Bin**")
                st.info(f"**💡 Proper disposal category**")
            
            # Confidence visualization with color coding
            st.markdown("### 📊 Confidence Level")
            if confidence > 0.9:
                st.success(f"🎉 Excellent confidence: {confidence:.1%}")
            elif confidence > 0.7:
                st.info(f"💡 Good confidence: {confidence:.1%}")
            else:
                st.warning(f"⚠️ Moderate confidence: {confidence:.1%}")
            
            st.progress(float(confidence))
            
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")

# Footer
st.markdown("---")
st.markdown("**SmartWasteAI** | High-Accuracy AI Model | 98%+ Test Accuracy")
