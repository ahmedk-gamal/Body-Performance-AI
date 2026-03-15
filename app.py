import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from streamlit_lottie import st_lottie
import requests
from fpdf import FPDF 
import json

# 1. إعدادات الصفحة بنمط Apple
st.set_page_config(
    page_title="Health AI | Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 Apple Style CSS (ألوان LinkedIn و GitHub والنبذة)
st.markdown("""
    <style>
    :root { --apple-red: #ff2d55; }
    .main { background-color: #f2f2f7; }
    
    /* ألوان الأزرار */
    .stButton>button { 
        width: 100%; border-radius: 12px; height: 3.5em; 
        background-color: #ff2d55; color: white; font-weight: 600; border: none; transition: 0.3s;
    }
    .stButton>button:hover { background-color: #e6264d; transform: translateY(-2px); }

    /* ألوان الروابط الأصلية */
    .linkedin-link { color: #0077b5 !important; font-weight: bold; text-decoration: none; display: flex; align-items: center; gap: 8px; font-size: 16px; }
    .github-link { color: #333 !important; font-weight: bold; text-decoration: none; display: flex; align-items: center; gap: 8px; font-size: 16px; }
    
    /* صندوق النبذة التقنية */
    .about-box {
        background-color: rgba(255, 45, 85, 0.05);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ff2d55;
        font-size: 14px;
        color: #1c1c1e;
        margin-bottom: 20px;
    }
    th { background-color: #ff2d55 !important; color: white !important; }
    [data-testid="stMetricValue"] { color: #ff2d55; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

# 2. دالة توليد التقرير PDF الاحترافي
def create_pdf(data_summary, plan_df):
    pdf = FPDF()
    pdf.add_page()
    
    # Sidebar line
    pdf.set_draw_color(255, 45, 85)
    pdf.set_line_width(1)
    pdf.line(65, 10, 65, 280)

    # Developer Info (Left Side)
    pdf.set_font("Arial", 'B', 14); pdf.set_text_color(255, 45, 85)
    pdf.text(10, 25, "HEALTH AI")
    
    pdf.set_font("Arial", 'B', 10); pdf.set_text_color(100, 100, 100)
    pdf.text(10, 40, "DEVELOPER:")
    pdf.set_text_color(0, 0, 0); pdf.set_font("Arial", '', 10)
    pdf.text(10, 47, "Ahmed Khaled Gamal")
    
    pdf.set_font("Arial", 'B', 10); pdf.set_text_color(100, 100, 100)
    pdf.text(10, 60, "CONTACT:")
    pdf.set_text_color(0, 0, 255); pdf.set_font("Arial", '', 9)
    pdf.text(10, 67, "LinkedIn: k-ahmed-auc")
    pdf.text(10, 74, "GitHub: ahmedk-gamal")

    # Main Content (Right Side)
    pdf.set_xy(70, 20)
    pdf.set_font("Arial", 'B', 22); pdf.set_text_color(255, 45, 85)
    pdf.cell(130, 15, "HEALTH ANALYSIS REPORT", ln=True)
    
    pdf.set_xy(70, 45); pdf.set_font("Arial", 'B', 12); pdf.set_text_color(0, 0, 0)
    pdf.cell(130, 10, "1. Biological Summary:", ln=True)
    
    pdf.set_font("Arial", '', 11)
    for key, value in data_summary.items():
        pdf.set_x(75); pdf.cell(130, 8, txt=f"- {key}: {value}", ln=True)
    
    pdf.ln(10); pdf.set_x(70); pdf.set_font("Arial", 'B', 12)
    pdf.cell(130, 10, "2. Weekly Fitness Plan:", ln=True)
    
    # Table headers
    pdf.set_font("Arial", 'B', 10); pdf.set_fill_color(255, 45, 85); pdf.set_text_color(255, 255, 255)
    pdf.set_x(70)
    pdf.cell(20, 10, "Day", 1, 0, 'C', True)
    pdf.cell(40, 10, "Workout", 1, 0, 'C', True)
    pdf.cell(35, 10, "Nutrition", 1, 0, 'C', True)
    pdf.cell(35, 10, "Advice", 1, 1, 'C', True)
    
    # Table data
    pdf.set_font("Arial", '', 9); pdf.set_text_color(0, 0, 0)
    for index, row in plan_df.iterrows():
        pdf.set_x(70)
        pdf.cell(20, 10, str(row[0]), 1)
        pdf.cell(40, 10, str(row[1]), 1)
        pdf.cell(35, 10, str(row[2]), 1)
        pdf.cell(35, 10, str(row[3]), 1)
        pdf.ln()
        
    return bytes(pdf.output())


# تابع للجزء الأول.. ضعه تحته مباشرة

# 3. تحميل الأصول (Assets)
def load_lottieurl(url):
    try: return requests.get(url, timeout=5).json()
    except: return None

lottie_heart = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_m6cu96.json")

@st.cache_resource
def load_assets():
    try:
        m = load_model('body_performance_model.keras')
        s = joblib.load('scaler.pkl')
        return m, s
    except: return None, None

model, scaler = load_assets()

# 4. السايد بار (الهوية البصرية والنبذة والروابط)
with st.sidebar:
    # 1. القلب النابض
    if lottie_heart:
        st_lottie(lottie_heart, height=160, key="heart_logo")
    else:
        st.title("❤️ Health AI")
    
    # 2. النبذة التقنية (Technical Brief)
    st.markdown("---")
    st.subheader("🤖 Technical Brief")
    st.markdown("""
    <div class="about-box">
    Built with a <b>Multi-Layer Perceptron (MLP)</b>, this AI predicts physical class based on biological metrics. 
    It features custom <b>Feature Engineering</b> (BMI & Pulse Pressure) to enhance decision-making accuracy.
    </div>
    """, unsafe_allow_html=True)
    
    # 3. بيانات المطور والروابط الملونة
    st.markdown("---")
    st.subheader("👨‍💻 Developer")
    st.markdown(f"**Ahmed Khaled Gamal**")
    st.markdown(f"""
    <a href="https://www.linkedin.com/in/k-ahmed-auc/" class="linkedin-link" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="20"> LinkedIn
    </a>
    <br>
    <a href="https://github.com/ahmedk-gamal" class="github-link" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="20"> GitHub
    </a>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    lang = st.radio("🌐 UI Language", ("English", "العربية"))

# 5. الواجهة الرئيسية واستمارة البيانات
st.title("🧬 Health AI Performance Analysis")
st.markdown("Predictive Health System | Powered by Deep Learning")

if model is None:
    st.error("Model or Scaler missing! Please upload files.")
    st.stop()

with st.form("health_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 10, 90, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height (cm)", 120.0, 220.0, 175.0)
        weight = st.number_input("Weight (kg)", 30.0, 200.0, 75.0)
        fat = st.number_input("Body Fat %", 5.0, 50.0, 20.0)
    with col2:
        sys = st.number_input("Systolic BP", 80, 200, 120)
        dia = st.number_input("Diastolic BP", 50, 120, 80)
        grip = st.number_input("Grip Force", 10.0, 100.0, 45.0)
        situps = st.number_input("Sit-ups Count", 0, 100, 35)
        jump = st.number_input("Broad Jump (cm)", 50.0, 350.0, 210.0)
        bend = st.number_input("Flexibility", -20.0, 50.0, 15.0)
    
    submit_btn = st.form_submit_button("Analyze Performance & Generate Report ✨")

# 6. معالجة النتائج وعرضها
if submit_btn:
    # 1. Feature Engineering
    gender_num = 0 if gender == "Male" else 1
    bmi_val = weight / ((height/100)**2)
    pp_val = sys - dia
    
    # 2. Prediction Pipeline
    raw_data = np.array([[age, gender_num, height, weight, fat, sys, dia, grip, situps, bend, jump, bmi_val, pp_val]])
    scaled_data = scaler.transform(raw_data)
    prediction = model.predict(scaled_data)
    class_idx = np.argmax(prediction)
    labels = ['A (Excellent)', 'B (Good)', 'C (Fair)', 'D (Poor)']
    
    # 3. Display Results
    st.markdown("---")
    res_c1, res_c2, res_c3 = st.columns(3)
    res_c1.metric("Class Result", labels[class_idx])
    res_c2.metric("Calculated BMI", f"{bmi_val:.2f}")
    res_c3.metric("Pulse Pressure", pp_val)

    # 4. Weekly Plan Table
    plan_df = pd.DataFrame({
        "Day": ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"],
        "Workout": ["Push Session", "Pull Session", "Leg Day", "Active Recovery", "HIIT Cardio", "Power Training", "Rest"],
        "Nutrition": ["High Protein", "High Protein", "Complex Carbs", "Vitamins Mix", "Low Carb", "Protein + Fiber", "Cheat Meal"],
        "Advice": ["Hydrate", "Deep Stretch", "8h Sleep", "Walk 30min", "Focus", "Intensity", "Full Rest"]
    })
    st.table(plan_df)
    
    # 5. PDF Generation & Download
    summary = {"Age": age, "BMI": f"{bmi_val:.2f}", "Pulse Pressure": pp_val, "Category": labels[class_idx]}
    pdf_final = create_pdf(summary, plan_df)
    
    st.download_button(
        label="📥 Download Full Health Report (PDF)",
        data=pdf_final,
        file_name=f"Health_Report_{age}.pdf",
        mime="application/pdf"
    )

st.caption("© 2026 Ahmed Khaled Gamal | Property & Tech Consultant")
