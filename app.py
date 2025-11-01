import streamlit as st
import numpy as np
from PIL import Image
import os
import urllib.request

# Set page configuration
st.set_page_config(
    page_title="SmartWasteAI - Waste Classification",
    page_icon="🗑️",
    layout="centered"
)

# Try to import TensorFlow with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    TENSORFLOW_AVAILABLE = False

# Define class labels
class_labels = {
    0: {'name': 'Biodegradable', 'bin_color': 'Green', 'bin_emoji': '🟢', 
        'bin_info': 'For organic waste like food scraps, garden waste.'},
    1: {'name': 'Hazardous', 'bin_color': 'Red', 'bin_emoji': '🔴',
        'bin_info': 'For dangerous materials like batteries, chemicals.'},
    2: {'name': 'Recyclable', 'bin_color': 'Blue', 'bin_emoji': '🔵',
        'bin_info': 'For materials that can be recycled like plastic, glass, metal.'}
}

@st.cache_resource
def load_waste_model():
    """
    Load the model with comprehensive error handling
    """
    if not TENSORFLOW_AVAILABLE:
        st.sidebar.error("❌ TensorFlow not available")
        return None
        
    try:
        # Check if model file exists
        model_path = "models/waste_classifier.h5"
        
        if not os.path.exists(model_path):
            st.sidebar.warning(f"⚠️ Model file not found at: {model_path}")
            st.sidebar.info("Running in demo mode with smart predictions")
            return None
        
        # Try to load the model
        model = load_model(model_path)
        st.sidebar.success("✅ AI Model loaded successfully!")
        return model
        
    except Exception as e:
        st.sidebar.error(f"❌ Model loading failed: {str(e)}")
        return None

# Load model
model = load_waste_model()

def preprocess_image(img):
    """Preprocess image for model prediction"""
    img = img.resize((224, 224))
    img_array = np.array(img)
    
    # If image is grayscale, convert to RGB
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA to RGB
        img_array = img_array[:, :, :3]
    
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

def predict_waste(img_array):
    """Make prediction with error handling"""
    if model is None or not TENSORFLOW_AVAILABLE:
        # Smart demo mode based on image analysis
        img_mean = np.mean(img_array)
        img_std = np.std(img_array)
        
        # Simple heuristic for demo purposes
        if img_std > 0.2:  # High variation - likely recyclable (mixed materials)
            return 2, 0.87  # Recyclable
        elif img_mean < 0.4:  # Dark image - likely hazardous
            return 1, 0.85  # Hazardous
        else:  # Medium brightness - likely biodegradable
            return 0, 0.83  # Biodegradable
    
    try:
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0, 0.5  # Default fallback

# Streamlit UI
st.title("♻️ SmartWasteAI")
st.markdown("### AI-Powered Waste Segregation System")

# Status indicator
if model is None:
    st.warning("🔧 **Running in Demo Mode** - Uploading a real model will enable AI predictions")
else:
    st.success("🤖 **AI Mode Active** - Real-time waste classification enabled")

st.write("Upload an image of waste to classify it into the correct disposal category.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image_display = Image.open(uploaded_file)
        st.image(image_display, caption="📸 Uploaded Image", use_column_width=True)
        
        # Preprocess and predict
        with st.spinner('🔍 Analyzing waste composition...'):
            processed_image = preprocess_image(image_display)
            predicted_class, confidence = predict_waste(processed_image)
        
        # Get class info
        class_info = class_labels[predicted_class]
        
        # Display results
        st.success("✅ Analysis Complete!")
        
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
            st.info(f"**💡 {class_info['bin_info']}**")
        
        # Confidence visualization
        st.markdown("### 📊 Confidence Level")
        st.progress(float(confidence))
        st.write(f"**Confidence Score:** {confidence:.1%}")
        
        # Demo mode notice
        if model is None:
            st.info("""
            **💡 Demo Mode Information:** 
            - Current predictions are simulated for demonstration
            - To enable real AI: Upload 'waste_classifier.h5' to the 'models' folder
            - File must be less than 25MB for GitHub deployment
            """)
            
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.info("💡 Try uploading a different image format (JPG, PNG)")

# Quick test examples
st.markdown("---")
st.markdown("### 🧪 Waste Examples")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🟢 Biodegradable**")
    st.write("- Food waste")
    st.write("- Paper products")
    st.write("- Garden waste")

with col2:
    st.markdown("**🔵 Recyclable**")
    st.write("- Plastic bottles")
    st.write("- Glass containers")
    st.write("- Metal cans")

with col3:
    st.markdown("**🔴 Hazardous**")
    st.write("- Batteries")
    st.write("- Electronics")
    st.write("- Chemicals")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About SmartWasteAI")
    st.write("""
    An intelligent waste classification system that helps 
    sort waste into proper disposal categories using AI.
    """)
    
    st.markdown("### 🗑️ Waste Categories")
    st.write("""
    **🟢 Green Bin - Biodegradable**
    - Food scraps
    - Yard waste
    - Paper products
    
    **🔵 Blue Bin - Recyclable**  
    - Plastics
    - Glass
    - Metals
    - Cardboard
    
    **🔴 Red Bin - Hazardous**
    - Batteries
    - Electronics
    - Chemicals
    - Medical waste
    """)
    
    st.markdown("---")
    st.header("🔧 System Status")
    
    status_color = "🟢" if model else "🟡"
    st.write(f"{status_color} **AI Model:** {'Loaded' if model else 'Demo Mode'}")
    st.write(f"🔧 **TensorFlow:** {'Available' if TENSORFLOW_AVAILABLE else 'Not Available'}")
    
    st.markdown("---")
    st.header("🎓 Educational Project")
    st.write("""
    **Machine Learning & Deep Learning Course**
    - AI in Action Project
    - Computer Vision Solution
    - Smart Waste Management
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p><b>SmartWasteAI</b> | Machine Learning & Deep Learning Course</p>
        <p>Developed for AI in Action: Solving Real-World Challenges</p>
    </div>
    """,
    unsafe_allow_html=True
)
