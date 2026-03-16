## 🏋️‍♂️ Body Performance AI: Deep Learning Classification & Advisory System



<p align="center">
<img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python" alt="Python">
<img src="https://img.shields.io/badge/TensorFlow-2.17-orange?style=for-the-badge&logo=tensorflow" alt="TensorFlow">
<img src="https://img.shields.io/badge/Streamlit-Live-red?style=for-the-badge&logo=streamlit" alt="Streamlit">
<img src="https://img.shields.io/badge/Maintained%3F-Yes-green?style=for-the-badge" alt="Maintained">
</p>


## 🌟 Overview
This project is an interactive AI platform designed for physical fitness assessment, developed as part of integrating data science into human performance analysis. The system analyzes biometric and athletic measurements to classify users into four performance tiers (A, B, C, D), while providing personalized health and exercise recommendations based on the results and Body Mass Index (BMI).


## 🏗️ Technical Architecture
🧠 Model Architecture
The core engine is a Multi-Layer Perceptron (MLP) deep neural network featuring:

Input Layer: Processes 13 features (including engineered domain-specific metrics).

Hidden Layers: Three dense layers (512 → 256 → 128 neurons) utilizing ReLU activation functions.

Regularization: Implemented Dropout (0.5) and Early Stopping to ensure model stability and prevent overfitting.

Optimizer: Adam optimizer paired with ReduceLROnPlateau scheduling to achieve optimal convergence and accuracy.


## 🧪 Feature Engineering
Model performance was enhanced through the innovation of additional physiological indicators:

BMI (Body Mass Index): Correlating weight and height to provide deeper physiological context.

Pulse Pressure: Calculating the delta between systolic and diastolic blood pressure as an indicator of cardiovascular efficiency.


## 💻 The Web App
The application is deployed via Streamlit, offering a seamless user experience:

User-friendly biometric and athletic data entry.

Instant classification of physical performance tiers.

Dynamic advisory generation based on the user's health status (Body Fat % and BMI).

🔗 Live Demo: Body Performance AI App https://body-performance-ai-e2avrvegv2mcxnso7bbsvu.streamlit.app/



## 🚀 Quick Start
Clone the Repository:

Bash
git clone https://github.com/ahmedk-gamal/body-performance-ai.git
cd body-performance-ai
Install Dependencies:

Bash
pip install -r requirements.txt
Run the Application:

Bash
streamlit run app.py


## 📊 Dataset
The model was trained on a comprehensive dataset of 13,393 records, including physical tests (Grip force, Sit-ups, Broad jump) and biometric measurements (Blood pressure, Body fat percentage).


## 🔮 Future Roadmap
[ ] Integration of full Arabic language support.

[ ] Leveraging Computer Vision for real-time exercise posture correction.

[ ] Developing a historical Dashboard to track user performance progress over time.


## 👨‍💼 Author
Ahmed Khaled Gamal Hossam Eldien

Machine Learning Engineer & Investment Consultant


## ⭐️ If you find this work helpful, please support the project with a Star on GitHub!
