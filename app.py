import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from streamlit_lottie import st_lottie
import requests
from fpdf import FPDF 
import base64

# 1. إعدادات الصفحة
st.set_page_config(page_title="Health AI | Apple Style", page_icon="❤️", layout="wide")

# 🎨 CSS المحسن (Apple Health Theme)
st.markdown("""
    <style>
    :root { --apple-red: #ff2d55; }
    .stButton>button { 
        width: 100%; border-radius: 12px; height: 3.5em; 
        background-color: #ff2d55; color: white; font-weight: 600; border: none;
    }
    .social-links a { text-decoration: none; color: inherit; display: flex; align-items: center; gap: 10px; padding: 5px; }
    th { background-color: #ff2d55 !important; color: white !important; }
    [data-testid="stMetricValue"] { color: #ff2d55; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

# 2. دالة توليد التقرير PDF (مع إضافة بيانات السايد بار داخل التقرير)
def create_pdf(t, data_summary, plan_df):
    pdf = FPDF()
    pdf.add_page()
    
    # رسم خط جانبي جمالي (Sidebar line in PDF)
    pdf.set_draw_color(255, 45, 85) # Apple Red
    pdf.set_line_width(1)
    pdf.line(65, 10, 65, 280)

    # --- القسم الجانبي في التقرير (Sidebar Section) ---
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(255, 45, 85)
    pdf.text(10, 20, "DEVELOPER")
    
    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(0, 0, 0)
    pdf.text(10, 30, "Ahmed Khaled Gamal")
    
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(255, 45, 85)
    pdf.text(10, 50, "CONTACT")
    pdf.set_font("Arial", '', 9)
    pdf.set_text_color(0, 0, 255)
    pdf.text(10, 60, "LinkedIn: k-ahmed-auc")
    pdf.text(10, 68, "GitHub: ahmedk-gamal")

    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(255, 45, 85)
    pdf.text(10, 90, "TECH STACK")
    pdf.set_font("Arial", '', 9)
    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(10, 95)
    pdf.multi_cell(50, 5, "Deep Learning (MLP)\nTensorFlow\nStreamlit UI")

    # --- القسم الرئيسي في التقرير (Main Content) ---
    pdf.set_xy(70, 20)
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(255, 45, 85)
    pdf.cell(130, 10, txt="HEALTH AI REPORT", ln=True, align='L')
    
    pdf.set_xy(70, 40)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(130, 10, txt="1. Biological Metrics Summary:", ln=True)
    
    pdf.set_font("Arial", '', 11)
    pdf.set_x(75)
    for key, value in data_summary.items():
        pdf.set_x(75)
        pdf.cell(130, 8, txt=f"- {key}: {value}", ln=True)
    
    pdf.ln(10)
    pdf.set_x(70)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(130, 10, txt="2. Weekly Personalized Plan:", ln=True)
    
    # جدول الخطة
    pdf.set_font("Arial", 'B', 9)
    pdf.set_fill_color(255, 45, 85)
    pdf.set_text_color(255, 255, 255)
    pdf.set_x(70)
    pdf.cell(20, 10, "Day", 1, 0, 'C', True)
    pdf.cell(40, 10, "Workout", 1, 0, 'C', True)
    pdf.cell(35, 10, "Nutrition", 1, 0, 'C', True)
    pdf.cell(35, 10, "Advice", 1, 1, 'C', True)
    
    pdf.set_font("Arial", '', 8)
    pdf.set_text_color(0, 0, 0)
    for index, row in plan_df.iterrows():
        pdf.set_x(70)
        pdf.cell(20, 10, str(row[0]), 1)
        pdf.cell(40, 10, str(row[1]), 1)
        pdf.cell(35, 10, str(row[2]), 1)
        pdf.cell(35, 10, str(row[3]), 1)
        pdf.ln()
        
    return bytes(pdf.output())

# 3. تحميل الأصول (القلب والموديل)
def load_lottieurl(url):
    try: return requests.get(url).json()
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

# 4. نظام اللغات والسايد بار
with st.sidebar:
    if lottie_heart: st_lottie(lottie_heart, height=150, key="heart_logo")
    lang = st.radio("🌐 Language / اللغة", ("English", "العربية"))
    st.markdown("---")

texts = {
    "English": {
        "title": "🧬 Health AI Analysis",
        "sub": "Predictive Health System inspired by Apple Health.",
        "about_h": "🤖 Tech Stack",
        "model_desc": "**Deep Learning (MLP)**\nNeural network with 3 hidden layers.",
        "dev_h": "👨‍💻 Developer",
        "btn": "Analyze & Generate Report ✨",
        "pdf_btn": "📥 Download Health Report (PDF)",
        "days": ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"],
        "classes": ['A (Excellent)', 'B (Good)', 'C (Fair)', 'D (Poor)']
    },
    "العربية": {
        "title": "🧬 تحليل المؤشرات الصحية الذكي",
        "sub": "نظام تنبؤ صحي متطور بتصميم Apple Health.",
        "about_h": "🤖 التقنيات",
        "model_desc": "**التعلم العميق (MLP)**\nشبكة عصبية لتصنيف الحالة البدنية.",
        "dev_h": "👨‍💻 المطور",
        "btn": "تحليل واستخراج التقرير ✨",
        "pdf_btn": "📥 تحميل التقرير الطبي (PDF)",
        "days": ["السبت", "الأحد", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة"],
        "classes": ['A (ممتاز)', 'B (جيد جداً)', 'C (جيد)', 'D (ضعيف)']
    }
}
t = texts[lang]

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

# 5. الواجهة الرئيسية
st.title(t["title"])
st.markdown(t["sub"])

if model is None:
    st.error("Missing Model Files!")
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
        bend = st.number_input("Flexibility", -20.0, 50.0, 15.0)
    submit = st.form_submit_button(t["btn"])

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

    # بيانات الخطة للجدول والتقرير
    plan_data = {
        "Day": ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"],
        "Workout": ["Push Day","Pull Day","Legs","Recovery","HIIT","Power","Rest"],
        "Nutrition": ["Berries","Broccoli","Avocado","Kiwi","Carrots","Pomegranate","Mix"],
        "Advice": ["Hydrate","Protein","Sleep","Fiber","Antiox","Repair","Relax"]
    }
    df_plan = pd.DataFrame(plan_data)
    st.table(df_plan)
    
    report_summary = {"Age": age, "Gender": gender, "BMI": f"{bmi:.2f}", "Fitness Class": t["classes"][idx]}
    pdf_bytes = create_pdf(t, report_summary, df_plan)
    
    st.download_button(label=t["pdf_btn"], data=pdf_bytes, file_name=f"Health_Report_{age}.pdf", mime="application/pdf")
