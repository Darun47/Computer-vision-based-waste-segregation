♻️ SmartWasteAI
https://static.streamlit.io/badges/streamlit_badge_black_white.svg
https://img.shields.io/badge/Python-3.8%252B-blue
https://img.shields.io/badge/TensorFlow-2.13%252B-orange

An AI-powered waste classification system that uses deep learning to sort waste into Biodegradable, Recyclable, and Hazardous categories.

🚀 Live Demo
Try it here!

Upload a waste image and get instant AI classification with disposal recommendations.

🎯 Features
AI Classification: Real-time waste sorting using MobileNetV2

Three Categories: 🟢 Biodegradable, 🔵 Recyclable, 🔴 Hazardous

High Accuracy: 93.8% validation accuracy

User-Friendly: Simple web interface built with Streamlit

Confidence Scores: Transparent probability metrics

🛠️ Installation
bash
# Clone repository
git clone https://github.com/Darun47/Computer-vision-based-waste-segregation.git
cd Computer-vision-based-waste-segregation

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
📦 Requirements
Python 3.8+

TensorFlow 2.13+

Streamlit 1.28+

Pillow 10.0+

🏗️ Project Structure
text
├── app.py                 # Main Streamlit application
├── waste_classifier.h5    # Trained model (25.1 MB)
├── requirements.txt       # Dependencies
└── README.md             # Documentation
🎓 Usage
Upload a waste image (JPG, PNG, JPEG)

Wait for AI analysis (2-3 seconds)

View classification results and disposal instructions

Follow the recommended bin color for proper disposal

🔬 Technical Details
Model: MobileNetV2 with transfer learning

Training: 95.7% training accuracy, 93.8% validation accuracy

Dataset: 10 waste categories, 70-20-10 split

Framework: TensorFlow/Keras + Streamlit

🌟 Future Enhancements
Expanded waste categories

Mobile app development

IoT smart bin integration

Multi-language support

👨‍💻 Author
Abhinav Chanakya
IBCP Student - Artificial Intelligence
Meluha International School

📄 License
MIT License - see LICENSE file for details.
