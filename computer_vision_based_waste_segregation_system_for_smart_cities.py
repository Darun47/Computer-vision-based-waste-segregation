import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="SmartWasteAI - Waste Classification",
    page_icon="🗑️",
    layout="centered"
)

# Debug info
st.sidebar.title("🔧 System Info")
st.sidebar.write(f"TensorFlow: {tf.__version__}")

# Load the trained model - CORRECT PATH FOR STREAMLIT CLOUD
@st.cache_resource
def load_waste_model():
    try:
        # For Streamlit Cloud deployment - model in models folder
        model_path = "models/waste_classifier.h5"
        
        # Check if model file exists
        if not os.path.exists(model_path):
            st.sidebar.error(f"❌ Model file not found at: {model_path}")
            st.sidebar.write("Current directory contents:")
            for item in os.listdir('.'):
                st.sidebar.write(f" - {item}")
            if os.path.exists('models'):
                st.sidebar.write("Models folder contents:")
                for item in os.listdir('models'):
                    st.sidebar.write(f" - {item}")
            return None
        
        model = load_model(model_path)
        st.sidebar.success("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        return None

# Load model
model = load_waste_model()

# Define class labels and their corresponding bin colors
class_labels = {
    0: {'name': 'Biodegradable', 'bin_color': 'Green', 'bin_emoji': '🟢', 
        'bin_info': 'For organic waste like food scraps, garden waste, paper, wood, and biodegradable materials.'},
    1: {'name': 'Hazardous', 'bin_color': 'Red', 'bin_emoji': '🔴',
        'bin_info': 'For dangerous materials like batteries, chemicals, medical waste, electronics, and toxic substances.'},
    2: {'name': 'Recyclable', 'bin_color': 'Blue', 'bin_emoji': '🔵',
        'bin_info': 'For materials that can be recycled like plastic, glass, metal, cardboard, and textiles.'}
}

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize to match model's expected input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create batch dimension
    img_array /= 255.0  # Rescale pixel values
    return img_array

# Function to make prediction
def predict_waste(img_array):
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
st.title("♻️ SmartWasteAI")
st.markdown("### AI-Powered Waste Segregation for Smart Cities")
st.write("Upload an image of a waste item, and our AI will classify it and recommend the correct disposal bin.")

# Demo mode notification
if model is None:
    st.warning("""
    ⚠️ **Demo Mode**: Model not loaded. 
    - For real AI classification, ensure 'waste_classifier.h5' is in the 'models' folder
    - Currently showing example predictions
    """)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image_display = Image.open(uploaded_file)
        st.image(image_display, caption="📸 Uploaded Image", use_column_width=True)
        
        # Preprocess and predict
        with st.spinner('🔍 AI is analyzing the waste item...'):
            if model is not None:
                # Real prediction with AI model
                processed_image = preprocess_image(image_display)
                predicted_class, confidence = predict_waste(processed_image)
            else:
                # Demo prediction (for testing without model)
                predicted_class = 1  # Default to Hazardous for demo
                confidence = 0.85
        
        if predicted_class is not None:
            # Get class info
            class_info = class_labels[predicted_class]
            
            # Display results
            st.success("✅ Analysis Complete!")
            
            # Create columns for layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="**Predicted Waste Type**", 
                    value=f"{class_info['bin_emoji']} {class_info['name']}"
                )
                st.metric(
                    label="**Confidence Level**", 
                    value=f"{confidence:.2%}"
                )
            
            with col2:
                # Display bin recommendation
                st.markdown(f"### 🗑️ Recommended Bin")
                st.markdown(f"# {class_info['bin_emoji']} **{class_info['bin_color']} Bin**")
                st.info(f"**💡 {class_info['bin_info']}**")
            
            # Confidence visualization
            st.markdown("### 📊 Confidence Level")
            st.progress(float(confidence))
            
            # Show detailed probabilities if model is loaded
            if model is not None:
                st.markdown("### 🔍 Detailed Analysis")
                predictions = model.predict(processed_image, verbose=0)[0]
                
                # Create columns for probability bars
                prob_col1, prob_col2, prob_col3 = st.columns(3)
                
                with prob_col1:
                    bio_prob = predictions[0]
                    st.metric("🟢 Biodegradable", f"{bio_prob:.2%}")
                    st.progress(float(bio_prob))
                
                with prob_col2:
                    haz_prob = predictions[1]
                    st.metric("🔴 Hazardous", f"{haz_prob:.2%}")
                    st.progress(float(haz_prob))
                
                with prob_col3:
                    rec_prob = predictions[2]
                    st.metric("🔵 Recyclable", f"{rec_prob:.2%}")
                    st.progress(float(rec_prob))
            
            # Success message based on confidence
            if confidence > 0.9:
                st.balloons()
                st.success("🎉 High confidence prediction! This waste item is clearly identified.")
            elif confidence > 0.7:
                st.info("💡 Good confidence level. The AI is fairly certain about this classification.")
            else:
                st.warning("⚠️ Lower confidence. Consider verifying this classification manually.")
                
        else:
            st.error("❌ Could not process the image. Please try another image.")
            
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.info("💡 Try uploading a different image format or size.")

# Quick test section
st.markdown("---")
st.markdown("### 🧪 Quick Test Examples")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🟢 Test Biodegradable"):
        st.info("Example: Food waste, garden trimmings, paper products")
        
with col2:
    if st.button("🔵 Test Recyclable"):
        st.info("Example: Plastic bottles, glass containers, metal cans")
        
with col3:
    if st.button("🔴 Test Hazardous"):
        st.info("Example: Batteries, electronics, chemicals")

# Add a comprehensive sidebar with information
with st.sidebar:
    st.markdown("---")
    st.header("ℹ️ About SmartWasteAI")
    st.write("""
    This intelligent system uses deep learning to automatically classify waste 
    into three categories for proper disposal and recycling.
    """)
    
    st.markdown("### 🗑️ Waste Categories")
    
    st.markdown("""
    **🟢 Green Bin - Biodegradable**
    - Food scraps & leftovers
    - Garden & yard waste  
    - Paper products
    - Wood materials
    - Organic materials
    """)
    
    st.markdown("""
    **🔵 Blue Bin - Recyclable**
    - Plastic containers
    - Glass bottles & jars
    - Metal cans & foil
    - Cardboard & paper
    - Textiles & clothes
    """)
    
    st.markdown("""
    **🔴 Red Bin - Hazardous**
    - Batteries (all types)
    - Electronics & e-waste
    - Chemicals & cleaners
    - Medical waste
    - Light bulbs
    - Paint & solvents
    """)
    
    st.markdown("---")
    st.markdown("### 🔧 Technical Details")
    st.write("""
    **Model Architecture:**
    - MobileNetV2 (Transfer Learning)
    - Custom classification layers
    - Trained on 6,700+ waste images
    
    **Performance:**
    - 85%+ accuracy on test data
    - Real-time classification
    - Confidence scoring
    """)
    
    st.markdown("---")
    st.markdown("### 🎓 Educational Project")
    st.write("""
    **Machine Learning & Deep Learning Course**
    - AI in Action: Solving Real-World Challenges
    - Computer Vision-based Waste Segregation
    - Smart City Solutions
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <h4>Developed as part of the <b>AI in Action Project</b></h4>
        <p><b>Machine Learning & Deep Learning Course</b> | Smart City Waste Management Solution</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Add custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
    }
    .stProgress > div > div > div > div {
        background-color: #2E8B57;
    }
    .success-message {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)
