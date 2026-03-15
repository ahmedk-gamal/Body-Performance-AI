import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from streamlit_lottie import st_lottie
import requests
from fpdf import FPDF # للتقرير
import base64

# 1. إعدادات الصفحة بنمط Apple Health
st.set_page_config(
    page_title="Health AI | Apple Style",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 Apple Health Style CSS
st.markdown("""
    <style>
    :root { --apple-red: #ff2d55; --apple-bg: #f2f2f7; }
    .main { background-color: var(--apple-bg); }
    .stButton>button { 
        width: 100%; border-radius: 12px; height: 3.5em; 
        background-color: #ff2d55; color: white; font-weight: 600; 
        font-size: 18px; border: none; transition: 0.3s;
    }
    .stButton>button:hover { background-color: #e6264d; transform: translateY(-2px); }
    .social-links a { text-decoration: none; color: inherit; display: flex; align-items: center; gap: 10px; padding: 5px; }
    th { background-color: #ff2d55 !important; color: white !important; text-align: center !important; }
    td { text-align: center !important; }
    [data-testid="stMetricValue"] { color: #ff2d55; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

# 2. دالة توليد تقرير PDF (English Only to avoid font issues)
def create_pdf(t, data_summary, plan_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Body Performance AI - Health Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="1. Biological Metrics Summary:", ln=True)
    pdf.set_font("Arial", '', 11)
    for key, value in data_summary.items():
        pdf.cell(200, 8, txt=f"- {key}: {value}", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="2. Weekly Fitness & Nutrition Plan:", ln=True)
    pdf.ln(5)
    
    # Table headers
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(25, 10, "Day", 1)
    pdf.cell(60, 10, "Workout", 1)
    pdf.cell(50, 10, "Nutrition", 1)
    pdf.cell(55, 10, "Advice", 1)
    pdf.ln()
    
    # Table rows
    pdf.set_font("Arial", '', 9)
    for index, row in plan_df.iterrows():
        pdf.cell(25, 10, str(row[0]), 1)
        pdf.cell(60, 10, str(row[1]), 1)
        pdf.cell(50, 10, str(row[2]), 1)
        pdf.cell(55, 10, str(row[3]), 1)
        pdf.ln()
        
    return pdf.output(dest='S').encode('latin-1')

# 3. تحميل الأصول (القلب والموديل)
def load_lottieurl(url):
    try: return requests.get(url).json()
    except: return None

lottie_heart = load_lottieurl("https://lottie.host/82334807-6c2e-4054-8e36-7c0a6b72a6b2/v8Y8yO5x20.json")

@st.cache_resource
def load_assets():
    try:
        m = load_model('body_performance_model.keras')
        s = joblib.load('scaler.pkl')
        return m, s
    except: return None, None

model, scaler = load_assets()

# 4. نظام اللغات
with st.sidebar:
    if lottie_heart: st_lottie(lottie_heart, height=140, key="heart")
    lang = st.radio("🌐 Language / اللغة", ("English", "العربية"))
    st.markdown("---")

texts = {
    "English": {
        "title": "🧬 Health AI Analysis",
        "sub": "Predictive Health System inspired by Apple Health.",
        "about_h": "🤖 Technology Stack",
        "model_desc": "**Deep Learning (MLP)**\nFully connected neural network for health classification.",
        "dev_h": "👨‍💻 Developer",
        "btn": "Analyze & Generate Report ✨",
        "pdf_btn": "📥 Download Health Report (PDF)",
        "days": ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"],
        "classes": ['A (Excellent)', 'B (Good)', 'C (Fair)', 'D (Poor)']
    },
    "العربية": {
        "title": "🧬 تحليل المؤشرات الصحية الذكي",
        "sub": "نظام تنبؤ صحي متطور بتصميم Apple Health.",
        "about_h": "🤖 التقنيات المستخدمة",
        "model_desc": "**التعلم العميق (MLP)**\nشبكة عصبية متطورة لتصنيف الحالة البدنية بدقة.",
        "dev_h": "👨‍💻 المطور",
        "btn": "تحليل واستخراج التقرير ✨",
        "pdf_btn": "📥 تحميل التقرير الطبي (PDF)",
        "days": ["السبت", "الأحد", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة"],
        "classes": ['A (ممتاز)', 'B (جيد جداً)', 'C (جيد)', 'D (ضعيف)']
    }
}
t = texts[lang]

# 5. عرض السايد بار
with st.sidebar:
    st.subheader(t["about_h"])
    st.info(t["model_desc"])
    st.markdown("---")
    st.subheader(t["dev_h"])
    st.markdown(f"**Ahmed Khaled Gamal**")
    st.markdown(f"""
    <div class="social-links">
        <a href="https://www.linkedin.com/in/k-ahmed-auc/" target="_blank"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="20"> LinkedIn</a>
        <a href="https://github.com/ahmedk-gamal" target="_blank"><img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="20"> GitHub</a>
    </div>
    """, unsafe_allow_html=True)

# 6. الواجهة الرئيسية والمدخلات
st.title(t["title"])
st.markdown(t["sub"])

if model is None:
    st.error("Model files not found! Ensure 'body_performance_model.keras' and 'scaler.pkl' are uploaded.")
    st.stop()

with st.form("health_form"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 10, 90, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height (cm)", 120.0, 220.0, 175.0)
        weight = st.number_input("Weight (kg)", 30.0, 200.0, 75.0)
        fat = st.number_input("Body Fat %", 5.0, 50.0, 20.0)
    with c2:
        sys = st.number_input("Systolic BP", 80, 200, 120)
        dia = st.number_input("Diastolic BP", 50, 120, 80)
        grip = st.number_input("Grip Force", 10.0, 100.0, 45.0)
        situps = st.number_input("Sit-ups Count", 0, 100, 35)
        jump = st.number_input("Broad Jump (cm)", 50.0, 350.0, 210.0)
        bend = st.number_input("Flexibility (Sit & Bend)", -20.0, 50.0, 15.0)
    
    submit = st.form_submit_button(t["btn"])

# 7. التوقع والنتائج
if submit:
    gender_val = 0 if gender == "Male" else 1
    bmi = weight / ((height / 100) ** 2)
    pp = sys - dia
    features = np.array([[age, gender_val, height, weight, fat, sys, dia, grip, situps, bend, jump, bmi, pp]])
    scaled = scaler.transform(features)
    pred = model.predict(scaled)
    idx = np.argmax(pred)
    
    st.markdown("---")
    r1, r2, r3 = st.columns(3)
    r1.metric("Category", t["classes"][idx])
    r2.metric("BMI", f"{bmi:.2f}")
    r3.metric("Pulse Pressure", pp)

    # جداول الخطة (English content for PDF compatibility)
    workouts = {0:["Push Day","Pull Day","Legs","Recovery","HIIT","Power","Rest"],
                1:["Chest/Tri","Back/Bi","Rest","Lower Body","Shoulders","Cardio","Rest"],
                2:["Power Walk","Rest","Bodyweight","Power Walk","Rest","Circuit","Active"],
                3:["Light Walk","Stretch","Rest","Light Walk","Yoga","Light Walk","Rest"]}
    nut = ["Berries","Broccoli","Avocado","Kiwi","Carrots","Pomegranate","Mix"]
    tips = ["High Carb","Protein","Hydrate","Complex Carbs","Antioxidants","Repair","Relax"]

    df_plan = pd.DataFrame({"Day": ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"], "Workout": workouts[idx], "Nutrition": nut, "Advice": tips})
    st.table(df_plan)
    
    # توليد وتحميل الـ PDF
    report_data = {"Age": age, "Gender": gender, "BMI": f"{bmi:.2f}", "Fitness Class": t["classes"][idx]}
    pdf_bytes = create_pdf(t, report_data, df_plan)
    
    st.download_button(label=t["pdf_btn"], data=pdf_bytes, file_name=f"Health_Report_{age}.pdf", mime="application/pdf")

st.caption("© 2026 Ahmed Khaled Gamal | All Rights Reserved")
