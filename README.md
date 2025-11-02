# Overview
♻️ SmartWasteAI

computer vision is used in smart waste sorting to separate trash into groups like Biodegredable, Recyclable and Hazardous. This system applies transfer learning using the architecture of MobileNetV2 for high accuracy with computational efficiency. The final solution is implemented in  the Streamlit web interface, to recognize real-time wastes and enable smart city waste management.

🚀Live Demo https://computer-vision-based-waste-segregation-vnewsecs6xzcxzc6g3bkdb.streamlit.app/

🎯Characteristics: AI Classification: MobileNetV2-based real-time garbage sorting

Three Types: 🟢 Biodegradable, 🔵 Recyclable, 🔴 Hazardous

High Accuracy: 93.8% validation accuracy

User-friendly: Streamlit was used to create a straight forward online interface.

Confidence Scores: Explicit measures of probability

🛠️ Installation?
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
Computer-vision-based-waste-segregation/
├── app.py                # Main Streamlit application

├── waste_classifier.h5   # Trained model (25.1 MB)

├── requirements.txt      # Python dependencies

├── .gitattributes        # Git LFS configuration

├── code_.txt             # Development notebooks

└── README.md             # Project documentation
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
S. DARUN 
1000309
IBCP Student - Artificial Intelligence
Meluha International School

📄 License
MIT License - see LICENSE file for details.
