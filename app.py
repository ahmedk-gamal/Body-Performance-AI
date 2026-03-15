import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from streamlit_lottie import st_lottie
import requests
from fpdf import FPDF 

# 1. إعدادات الصفحة
st.set_page_config(page_title="Health AI | Dashboard", page_icon="❤️", layout="wide")

# 🎨 CSS الاحترافي (ألوان النبذة الجديدة + التنسيق العام)
st.markdown("""
    <style>
    :root { --apple-red: #ff2d55; --soft-blue: #e3f2fd; --deep-blue: #1565c0; }
    [data-testid="stSidebar"] { background-color: #ffffff; }
    
    /* تصميم الأزرار */
    .stButton>button { 
        width: 100%; border-radius: 12px; height: 3.5em; 
        background-color: var(--apple-red); color: white; font-weight: 600; border: none; transition: 0.3s;
    }
    .stButton>button:hover { background-color: #e6264d; transform: translateY(-2px); }

    /* ألوان الروابط */
    .linkedin-link { color: #0077b5 !important; font-weight: bold; text-decoration: none; display: flex; align-items: center; gap: 8px; }
    .github-link { color: #333 !important; font-weight: bold; text-decoration: none; display: flex; align-items: center; gap: 8px; }
    
    /* صندوق النبذة التقنية (الألوان الزرقاء المريحة اللي طلبتها) */
    .about-box {
        background-color: var(--soft-blue); padding: 15px; border-radius: 10px;
        border-left: 5px solid var(--deep-blue); font-size: 14px; color: #0d47a1; margin-bottom: 20px;
    }
    
    th { background-color: var(--apple-red) !important; color: white !important; }
    [data-testid="stMetricValue"] { color: var(--apple-red); font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

# 2. دالة توليد التقرير PDF (يدعم اللغتين)
def create_pdf(t, data_summary, plan_df):
    pdf = FPDF()
    pdf.add_page()
    # تصميم التقرير
    pdf.set_draw_color(255, 45, 85); pdf.set_line_width(1); pdf.line(65, 10, 65, 280)
    pdf.set_font("Arial", 'B', 14); pdf.set_text_color(255, 45, 85); pdf.text(10, 25, "HEALTH AI")
    pdf.set_font("Arial", 'B', 10); pdf.set_text_color(0, 0, 0); pdf.text(10, 40, "DEVELOPER:")
    pdf.set_font("Arial", '', 10); pdf.text(10, 47, "Ahmed Khaled Gamal")
    
    pdf.set_xy(70, 20); pdf.set_font("Arial", 'B', 20); pdf.cell(130, 10, "HEALTH ANALYSIS REPORT", ln=True)
    pdf.ln(10); pdf.set_font("Arial", '', 11)
    for key, value in data_summary.items():
        pdf.set_x(70); pdf.cell(130, 8, txt=f"- {key}: {value}", ln=True)
    return bytes(pdf.output())

# (تابع للجزء الأول..)

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

# 4. قاموس اللغات الشامل (The Core)
texts = {
    "English": {
        "title": "🧬 Health AI Analysis",
        "brief_h": "🤖 Technical Brief",
        "brief_txt": "Built with a <b>Multi-Layer Perceptron (MLP)</b>, this AI predicts physical class based on biological metrics. It features custom <b>Feature Engineering</b> (BMI & Pulse Pressure).",
        "dev_h": "👨‍💻 Developer",
        "btn": "Analyze & Generate Report ✨",
        "pdf_btn": "📥 Download Health Report (PDF)",
        "labels": ["Age", "Gender", "Height (cm)", "Weight (kg)", "Fat %", "Systolic BP", "Diastolic BP", "Grip Force", "Sit-ups", "Jump (cm)", "Flexibility"],
        "res_h": "📊 Results",
        "cat": "Category",
        "bmi_lab": "BMI",
        "classes": ['A (Excellent)', 'B (Good)', 'C (Fair)', 'D (Poor)'],
        "days": ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"],
        "workouts": ["Push Day", "Pull Day", "Legs Day", "Recovery", "HIIT", "Power", "Rest"],
        "nutrition": ["High Protein", "High Protein", "Carb Load", "Vitamins", "Low Carb", "Fiber", "Cheat"],
        "advice": ["Hydrate", "Stretch", "Sleep", "Walk", "Focus", "Intensity", "Relax"]
    },
    "العربية": {
        "title": "🧬 تحليل المؤشرات الصحية بالذكاء الاصطناعي",
        "brief_h": "🤖 نبذة تقنية",
        "brief_txt": "تم بناء هذا النظام باستخدام <b>الشبكات العصبية (MLP)</b> للتنبؤ بمستوى الأداء البدني، مع دمج <b>هندسة البيانات</b> (BMI وضغط النبض).",
        "dev_h": "👨‍💻 المطور",
        "btn": "تحليل واستخراج التقرير ✨",
        "pdf_btn": "📥 تحميل التقرير الطبي (PDF)",
        "labels": ["العمر", "الجنس", "الطول (سم)", "الوزن (كجم)", "نسبة الدهون %", "ضغط الانقباض", "ضغط الانبساط", "قوة القبضة", "تمارين البطن", "الوثب", "المرونة"],
        "res_h": "📊 النتائج",
        "cat": "تصنيف اللياقة",
        "bmi_lab": "مؤشر الكتلة",
        "classes": ['A (ممتاز)', 'B (جيد جداً)', 'C (متوسط)', 'D (ضعيف)'],
        "days": ["السبت", "الأحد", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة"],
        "workouts": ["تمارين دفع", "تمارين سحب", "تمارين أرجل", "استشفاء", "كارديو", "تمارين قوة", "راحة تامة"],
        "nutrition": ["بروتين عالي", "بروتين عالي", "تحميل كارب", "فيتامينات", "كارب منخفض", "ألياف", "وجبة مفتوحة"],
        "advice": ["شرب ماء", "إطالات", "نوم كافٍ", "مشي", "تركيز", "كثافة", "راحة"]
    }
}

# 5. السايد بار (القلب + النبذة + الروابط)
with st.sidebar:
    if lottie_heart: st_lottie(lottie_heart, height=150, key="heart")
    lang = st.radio("🌐 Language / اللغة", ("English", "العربية"))
    t = texts[lang]
    
    st.markdown("---")
    st.subheader(t["brief_h"])
    st.markdown(f'<div class="about-box">{t["brief_txt"]}</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.subheader(t["dev_h"])
    st.markdown(f"**Ahmed Khaled Gamal**")
    st.markdown(f"""
        <a href="https://www.linkedin.com/in/k-ahmed-auc/" class="linkedin-link" target="_blank">LinkedIn</a><br>
        <a href="https://github.com/ahmedk-gamal" class="github-link" target="_blank">GitHub</a>
    """, unsafe_allow_html=True)


# (تابع للجزء الثاني..)

# 6. الواجهة الرئيسية
st.title(t["title"])
if model is None: st.error("Model missing!"); st.stop()

with st.form("main_form"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input(t["labels"][0], 10, 90, 25)
        gender = st.selectbox(t["labels"][1], ["Male", "Female"] if lang == "English" else ["ذكر", "أنثى"])
        height = st.number_input(t["labels"][2], 120.0, 220.0, 175.0)
        weight = st.number_input(t["labels"][3], 30.0, 200.0, 75.0)
        fat = st.number_input(t["labels"][4], 5.0, 50.0, 20.0)
    with c2:
        sys = st.number_input(t["labels"][5], 80, 200, 120); dia = st.number_input(t["labels"][6], 50, 120, 80)
        grip = st.number_input(t["labels"][7], 10.0, 100.0, 45.0); situps = st.number_input(t["labels"][8], 0, 100, 35)
        jump = st.number_input(t["labels"][9], 50.0, 350.0, 210.0); bend = st.number_input(t["labels"][10], -20.0, 50.0, 15.0)
    submit = st.form_submit_button(t["btn"])

if submit:
    # منطق التوقع (Prediction Logic)
    g_val = 0 if (gender == "Male" or gender == "ذكر") else 1
    bmi = weight / ((height/100)**2); pp = sys - dia
    features = np.array([[age, g_val, height, weight, fat, sys, dia, grip, situps, bend, jump, bmi, pp]])
    scaled = scaler.transform(features)
    idx = np.argmax(model.predict(scaled))
    
    # النتائج
    st.markdown("---")
    st.subheader(t["res_h"])
    r1, r2, r3 = st.columns(3)
    r1.metric(t["cat"], t["classes"][idx]); r2.metric(t["bmi_lab"], f"{bmi:.2f}"); r3.metric("Pulse Pressure", pp)
    
    # الجدول المترجم
    plan_df = pd.DataFrame({
        "Day / اليوم": t["days"], "Workout / التمرين": t["workouts"],
        "Nutrition / التغذية": t["nutrition"], "Advice / نصيحة": t["advice"]
    })
    st.table(plan_df)
    
    # التقرير
    pdf_bytes = create_pdf(t, {t["labels"][0]: age, "BMI": f"{bmi:.2f}", t["cat"]: t["classes"][idx]}, plan_df)
    st.download_button(t["pdf_btn"], data=pdf_bytes, file_name="Report.pdf")

st.caption("© 2026 Ahmed Khaled Gamal | Property & Tech Consultant")
