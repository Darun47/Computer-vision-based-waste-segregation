import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os
from model_architecture import create_smartwaste_model, preprocess_image, predict_waste

# Page configuration
st.set_page_config(
    page_title="SmartWasteAI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .recyclable {
        background-color: #90EE90;
        border: 2px solid #2E8B57;
    }
    .non-recyclable {
        background-color: #FFB6C1;
        border: 2px solid #DC143C;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">♻️ SmartWasteAI</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Waste Classification System")
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", 
                                   ["Home", "Image Classification", "About"])
    
    if app_mode == "Home":
        show_home()
    elif app_mode == "Image Classification":
        show_classification()
    else:
        show_about()

def show_home():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to SmartWasteAI!
        
        **SmartWasteAI** is an intelligent waste classification system that helps:
        
        - 🏠 **Households** properly sort their waste
        - 🏢 **Businesses** improve recycling efficiency
        - 🌍 **Communities** reduce contamination in recycling streams
        
        ### How it works:
        1. Upload an image of waste item
        2. Our AI model analyzes the image
        3. Get instant classification: Recyclable or Non-Recyclable
        4. Receive proper disposal guidance
        
        ### Supported Waste Types:
        - Plastic bottles & containers
        - Paper & cardboard
        - Glass items
        - Metal cans
        - Organic waste
        - Electronic waste
        - And more!
        """)
    
    with col2:
        st.image("https://images.unsplash.com/photo-1587334894137-85e7befb7597?w=400", 
                 caption="Smart Waste Management")
        st.info("💡 **Tip**: Take clear, well-lit photos for best results!")

def show_classification():
    st.markdown("## 🖼️ Waste Classification")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload an image of waste item", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Supported formats: JPG, JPEG, PNG, BMP"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process image
            if st.button("🔍 Analyze Waste", type="primary"):
                with st.spinner("Analyzing waste type..."):
                    try:
                        # Preprocess and predict
                        processed_image = preprocess_image(image)
                        prediction, confidence = predict_waste(processed_image)
                        
                        # Display results
                        with col2:
                            st.markdown("## 📊 Analysis Results")
                            
                            # Confidence gauge
                            st.metric("Confidence", f"{confidence:.2%}")
                            
                            # Prediction box
                            if prediction == "Recyclable":
                                st.markdown(
                                    f'<div class="prediction-box recyclable">'
                                    f'<h2>♻️ RECYCLABLE</h2>'
                                    f'<p>This item can be recycled!</p>'
                                    f'</div>', 
                                    unsafe_allow_html=True
                                )
                                
                                st.markdown("""
                                ### ✅ Proper Disposal:
                                - Clean and dry the item
                                - Place in recycling bin
                                - Remove any non-recyclable parts
                                """)
                            else:
                                st.markdown(
                                    f'<div class="prediction-box non-recyclable">'
                                    f'<h2>🚫 NON-RECYCLABLE</h2>'
                                    f'<p>This item should go in general waste</p>'
                                    f'</div>', 
                                    unsafe_allow_html=True
                                )
                                
                                st.markdown("""
                                ### 🗑️ Proper Disposal:
                                - Place in general waste bin
                                - Consider alternatives to reduce waste
                                - Check local guidelines for special disposal
                                """)
                        
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
                        st.info("Please try another image or check the file format.")

def show_about():
    st.markdown("""
    ## About SmartWasteAI
    
    **SmartWasteAI** is built using state-of-the-art deep learning technology 
    to help solve the global waste management challenge.
    
    ### Technology Stack:
    - **Framework**: TensorFlow/Keras
    - **Frontend**: Streamlit
    - **Computer Vision**: Convolutional Neural Networks
    - **Deployment**: Streamlit Cloud/Heroku
    
    ### Model Features:
    - High accuracy waste classification
    - Real-time processing
    - Support for multiple image formats
    - Scalable architecture
    
    ### Environmental Impact:
    By helping people properly sort waste, we aim to:
    - Increase recycling rates
    - Reduce contamination in recycling streams
    - Promote sustainable waste management practices
    
    ---
    *Built with ❤️ for a cleaner planet*
    """)

if __name__ == "__main__":
    main()
