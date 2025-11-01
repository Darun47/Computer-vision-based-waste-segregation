import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import requests


# Set page configuration
st.set_page_config(
    page_title="SmartWasteAI - Waste Classification System",
    page_icon="🗑️",
    layout="centered"
)

# ========== ADD YOUR MODEL LOADING CODE HERE ==========
@st.cache_resource
def load_real_model():
    """
    Load your actual trained model
    Replace the model path with your actual model file location
    """
    try:
        # OPTION 1: If model is in your Google Drive (mounted in Colab)
        model_path = "/content/drive/MyDrive/DATA./model/waste_classifier (1).tflite.py (1).py"
        
        # OPTION 2: If you uploaded model to Streamlit cloud
        # model_path = "waste_classifier.h5"
        
        # OPTION 3: Download from Google Drive
        # model_path = "downloaded_model.h5"
        # if not os.path.exists(model_path):
        #     gdown.download('https://drive.google.com/uc?id=YOUR_FILE_ID', model_path, quiet=False)
        
        if os.path.exists(model_path):
            model = load_model(model_path)
            st.sidebar.success("✅ Real AI Model Loaded Successfully!")
            return model
        else:
            st.sidebar.warning("⚠️ Model file not found. Using demo mode.")
            return None
            
    except Exception as e:
        st.sidebar.error(f"❌ Model loading failed: {str(e)}")
        st.sidebar.info("💡 Using demo mode temporarily")
        return None

# Load the model
model = load_real_model()
# ========== END OF MODEL LOADING CODE ==========

# Define class labels
class_labels = {
    0: {'name': 'Biodegradable', 'bin_color': 'Green', 'bin_emoji': '🟢', 
        'bin_info': 'For organic waste like food scraps, garden waste, paper products.'},
    1: {'name': 'Hazardous', 'bin_color': 'Red', 'bin_emoji': '🔴',
        'bin_info': 'For dangerous materials like batteries, chemicals, electronics, medical waste.'},
    2: {'name': 'Recyclable', 'bin_color': 'Blue', 'bin_emoji': '🔵',
        'bin_info': 'For materials that can be recycled like plastic, glass, metal, cardboard.'}
}

def preprocess_image(img):
    """Preprocess image exactly like during training"""
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_waste_demo(img_array):
    """Demo prediction function - used when real model isn't available"""
    # Simple rule-based demo prediction
    img_for_demo = img_array[0]  # Remove batch dimension
    
    # Calculate average brightness (simple heuristic)
    avg_brightness = np.mean(img_for_demo)
    
    # Simple rules for demo
    if avg_brightness < 0.3:
        return 1, 0.85  # Dark image -> hazardous
    elif avg_brightness > 0.7:
        return 2, 0.88  # Bright image -> recyclable
    else:
        return 0, 0.82  # Medium brightness -> biodegradable

def predict_waste(img_array):
    """Make prediction - tries real model first, falls back to demo"""
    global model
    
    if model is not None:
        try:
            predictions = model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            return predicted_class, confidence
        except Exception as e:
            st.sidebar.warning(f"Model prediction failed, using demo mode: {e}")
    
    # Fallback to demo mode
    return predict_waste_demo(img_array)

# Streamlit UI
st.title("♻️ SmartWasteAI")
st.markdown("### AI-Powered Waste Classification System")

# Model status
st.sidebar.header("🔧 System Status")
if model:
    st.sidebar.success("🤖 AI Model: **Loaded**")
    st.sidebar.info("🎯 Mode: **High-Accuracy AI**")
else:
    st.sidebar.warning("🤖 AI Model: **Demo Mode**")
    st.sidebar.info("💡 Upload images to test the system")

st.write("Upload an image of waste for AI-powered classification and sorting recommendations.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image_display = Image.open(uploaded_file)
        st.image(image_display, caption="📸 Uploaded Image", use_column_width=True)
        
        # Preprocess and predict
        with st.spinner('🔍 AI analysis in progress...'):
            processed_image = preprocess_image(image_display)
            predicted_class, confidence = predict_waste(processed_image)
        
        class_info = class_labels[predicted_class]
        
        # Display results
        st.success("✅ **Analysis Complete!**")
        
        # Results layout
        col1, col2, col3 = st.columns([1,1,1])
        
        with col1:
            st.metric(
                label="**Waste Type**", 
                value=f"{class_info['bin_emoji']} {class_info['name']}"
            )
        
        with col2:
            st.metric(
                label="**Confidence**", 
                value=f"{confidence:.1%}"
            )
        
        with col3:
            st.markdown(f"### 🗑️ {class_info['bin_color']} Bin")
        
        # Bin information
        st.info(f"**Disposal Instructions:** {class_info['bin_info']}")
        
        # Confidence indicator
        st.markdown("### 📊 Confidence Level")
        if confidence > 0.90:
            st.success(f"**Excellent Confidence: {confidence:.1%}** - Highly reliable")
        elif confidence > 0.80:
            st.info(f"**Good Confidence: {confidence:.1%}** - Reliable prediction")
        elif confidence > 0.70:
            st.warning(f"**Moderate Confidence: {confidence:.1%}** - Acceptable")
        else:
            st.error(f"**Low Confidence: {confidence:.1%}** - Manual check recommended")
        
        st.progress(float(confidence))
        
        # Show detailed probabilities if using real model
        if model:
            st.markdown("### 🔍 Detailed Analysis")
            predictions = model.predict(processed_image, verbose=0)[0]
            
            prob_col1, prob_col2, prob_col3 = st.columns(3)
            
            with prob_col1:
                bio_prob = predictions[0]
                st.metric("🟢 Biodegradable", f"{bio_prob:.2%}")
            
            with prob_col2:
                haz_prob = predictions[1]
                st.metric("🔴 Hazardous", f"{haz_prob:.2%}")
            
            with prob_col3:
                rec_prob = predictions[2]
                st.metric("🔵 Recyclable", f"{rec_prob:.2%}")
        else:
            st.warning("""
            **💡 Demo Mode Active** 
            - Real model would provide higher accuracy
            - Upload different images to test the interface
            """)
            
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")

# System information
st.markdown("---")
st.markdown("### 🏢 SmartWasteAI - Waste Classification System")
st.markdown("""
**🎯 Features:**
- AI-powered waste classification
- Smart bin recommendations  
- Educational waste management
- Real-time processing

**🔧 Technology:**
- TensorFlow Deep Learning
- Computer Vision
- MobileNetV2 Architecture
- Streamlit Web Interface
""")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
        <h3>🎓 Academic Project - SmartWasteAI</h3>
        <p><b>Machine Learning & Deep Learning Course</b></p>
        <p>AI-Powered Waste Classification System</p>
    </div>
    """,
    unsafe_allow_html=True
)
