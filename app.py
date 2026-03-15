import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from streamlit_lottie import st_lottie
import requests
from fpdf import FPDF 

# 1. إعدادات الصفحة بنمط Apple
st.set_page_config(
    page_title="Health AI | Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CSS الاحترافي (شامل ألوان النبذة والروابط وتنسيق الجداول)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    
    :root { 
        --apple-red: #ff2d55; 
        --soft-blue: #f0f7ff; 
        --deep-blue: #1a237e; 
    }
    
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }
    
    .main { background-color: #f2f2f7; }

    /* تنسيق الأزرار */
    .stButton>button { 
        width: 100%; 
        border-radius: 12px; 
        height: 3.8em; 
        background-color: var(--apple-red); 
        color: white; 
        font-weight: 700; 
        border: none; 
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover { 
        background-color: #e6264d; 
        transform: translateY(-2px); 
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }

    /* صندوق النبذة التقنية (الأزرق المريح) */
    .about-box {
        background-color: var(--soft-blue); 
        padding: 20px; 
        border-radius: 15px;
        border-left: 6px solid #2196F3; 
        font-size: 14px; 
        color: var(--deep-blue); 
        margin-bottom: 25px;
        line-height: 1.6;
    }

    /* تنسيق الروابط الملونة */
    .linkedin-link { color: #0077b5 !important; font-weight: bold; text-decoration: none; display: flex; align-items: center; gap: 10px; font-size: 16px; margin-bottom: 10px; }
    .github-link { color: #333 !important; font-weight: bold; text-decoration: none; display: flex; align-items: center; gap: 10px; font-size: 16px; }
    
    /* تنسيق الجداول والـ Metrics */
    th { background-color: var(--apple-red) !important; color: white !important; text-align: center !important; }
    [data-testid="stMetricValue"] { color: var(--apple-red); font-weight: 800; font-size: 2rem; }
    [data-testid="stMetricLabel"] { font-weight: 600; color: #48484a; }
    </style>
    """, unsafe_allow_html=True)

# 3. دالة توليد التقرير PDF
def create_pdf(t, summary, plan):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(255, 45, 85)
    pdf.rect(0, 0, 60, 300, 'F') # شريط جانبي ملون
    
    # بيانات المطور (يسار)
    pdf.set_font("Arial", 'B', 16); pdf.set_text_color(255, 255, 255)
    pdf.text(10, 30, "HEALTH AI")
    pdf.set_font("Arial", '', 10); pdf.text(10, 50, "DEVELOPER:")
    pdf.set_font("Arial", 'B', 10); pdf.text(10, 56, "Ahmed Khaled Gamal")
    
    # المحتوى (يمين)
    pdf.set_xy(70, 30); pdf.set_text_color(0, 0, 0); pdf.set_font("Arial", 'B', 22)
    pdf.cell(130, 10, "ANALYSIS REPORT", ln=True)
    pdf.ln(15)
    
    pdf.set_font("Arial", 'B', 12)
    for k, v in summary.items():
        pdf.set_x(70); pdf.cell(40, 10, f"{k}:", 0); pdf.set_font("Arial", '', 12)
        pdf.cell(90, 10, f"{v}", ln=True); pdf.set_font("Arial", 'B', 12)
        
    return bytes(pdf.output())

# (تابع للجزء الأول..)

# 4. تحميل الأصول والذكاء الاصطناعي
def load_lottie(url):
    try: return requests.get(url).json()
    except: return None

heart_anim = load_lottie("https://assets5.lottiefiles.com/packages/lf20_m6cu96.json")

@st.cache_resource
def get_models():
    try:
        m = load_model('body_performance_model.keras')
        s = joblib.load('scaler.pkl')
        return m, s
    except: return None, None

model, scaler = get_models()

# 5. القاموس الضخم (سر الـ Multi-language)
texts = {
    "English": {
        "title": "🧬 Health AI | Performance Dashboard",
        "brief_h": "🤖 Technical Brief",
        "brief_txt": "This system utilizes a <b>Deep Learning MLP</b> model to classify physical performance. It incorporates <b>Feature Engineering</b> like BMI and Pulse Pressure to provide higher diagnostic accuracy.",
        "dev_h": "👨‍💻 Developer Information",
        "btn": "Run AI Analysis & Export Report ✨",
        "pdf_btn": "📥 Download Official PDF Report",
        "labels": ["Age", "Gender", "Height (cm)", "Weight (kg)", "Body Fat %", "Systolic BP", "Diastolic BP", "Grip Force", "Sit-ups", "Broad Jump", "Flexibility"],
        "res_h": "📊 Comprehensive Analytics",
        "cat": "Performance Category",
        "bmi_lab": "BMI Score",
        "pp_lab": "Pulse Pressure",
        "classes": ['A (Excellent)', 'B (Good)', 'C (Fair)', 'D (Poor)'],
        "days": ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"],
        "workouts": ["Push Session", "Pull Session", "Legs Session", "Active Recovery", "HIIT Cardio", "Power Training", "Rest Day"],
        "nutrition": ["High Protein", "High Protein", "Carb Load", "Vitamins", "Low Carb", "Protein+Fiber", "Cheat Meal"],
        "advice": ["Hydrate", "Stretch", "8h Sleep", "Walk 30m", "Focus", "Intensity", "Full Rest"]
    },
    "العربية": {
        "title": "🧬 لوحة تحكم الذكاء الاصطناعي للصحة",
        "brief_h": "🤖 نبذة تقنية للمشروع",
        "brief_txt": "يعتمد النظام على نموذج <b>Deep Learning (MLP)</b> للتنبؤ بمستوى الأداء البدني. تم دمج <b>هندسة البيانات</b> لتحسين النتائج عبر حساب مؤشر كتلة الجسم وضغط النبض آلياً.",
        "dev_h": "👨‍💻 معلومات المطور",
        "btn": "تشغيل التحليل واستخراج التقرير ✨",
        "pdf_btn": "📥 تحميل التقرير الرسمي (PDF)",
        "labels": ["العمر", "الجنس", "الطول (سم)", "الوزن (كجم)", "نسبة الدهون %", "الضغط الانقباضي", "الضغط الانبساطي", "قوة القبضة", "تمارين البطن", "الوثب العريض", "المرونة"],
        "res_h": "📊 التحليلات الشاملة",
        "cat": "فئة الأداء البدني",
        "bmi_lab": "مؤشر كتلة الجسم",
        "pp_lab": "ضغط النبض",
        "classes": ['A (ممتاز)', 'B (جيد جداً)', 'C (متوسط)', 'D (ضعيف)'],
        "days": ["السبت", "الأحد", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة"],
        "workouts": ["تمارين الدفع", "تمارين السحب", "تمارين الأرجل", "استشفاء نشط", "كارديو HIIT", "تمارين القوة", "يوم راحة"],
        "nutrition": ["بروتين عالي", "بروتين عالي", "تحميل كربوهيدرات", "فيتامينات", "كربوهيدرات منخفضة", "بروتين + ألياف", "وجبة مفتوحة"],
        "advice": ["شرب الماء", "إطالات", "نوم 8 ساعات", "مشي 30 دقيقة", "تركيز عالي", "كثافة تدريب", "استشفاء"]
    }
}

# 6. السايد بار
with st.sidebar:
    if heart_anim: st_lottie(heart_anim, height=150, key="heart")
    lang = st.radio("🌐 Change Language / تغيير اللغة", ("English", "العربية"))
    t = texts[lang]
    
    st.markdown("---")
    st.subheader(t["brief_h"])
    st.markdown(f'<div class="about-box">{t["brief_txt"]}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader(t["dev_h"])
    st.markdown(f"**Ahmed Khaled Gamal**\n\nProperty & Tech Consultant")
    st.markdown(f"""
        <a href="https://www.linkedin.com/in/k-ahmed-auc/" class="linkedin-link" target="_blank">LinkedIn</a>
        <a href="https://github.com/ahmedk-gamal" class="github-link" target="_blank">GitHub</a>
    """, unsafe_allow_html=True)

# 7. جسم التطبيق
st.title(t["title"])
if model is None: st.error("Model files missing!"); st.stop()

with st.form("health_ai_form"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input(t["labels"][0], 10, 90, 25)
        gender = st.selectbox(t["labels"][1], ["Male", "Female"] if lang == "English" else ["ذكر", "أنثى"])
        h = st.number_input(t["labels"][2], 120.0, 220.0, 175.0)
        w = st.number_input(t["labels"][3], 30.0, 200.0, 75.0)
        f = st.number_input(t["labels"][4], 5.0, 50.0, 20.0)
    with c2:
        s_bp = st.number_input(t["labels"][5], 80, 200, 120)
        d_bp = st.number_input(t["labels"][6], 50, 120, 80)
        grp = st.number_input(t["labels"][7], 10.0, 100.0, 45.0)
        stp = st.number_input(t["labels"][8], 0, 100, 35)
        jmp = st.number_input(t["labels"][9], 50.0, 350.0, 210.0)
        bnd = st.number_input(t["labels"][10], -20.0, 50.0, 15.0)
    
    run_btn = st.form_submit_button(t["btn"])

if run_btn:
    # المعالجة
    g_num = 0 if (gender in ["Male", "ذكر"]) else 1
    bmi_v = w / ((h/100)**2); pp_v = s_bp - d_bp
    
    inputs = scaler.transform([[age, g_num, h, w, f, s_bp, d_bp, grp, stp, bnd, jmp, bmi_v, pp_v]])
    res_idx = np.argmax(model.predict(inputs))
    
    st.markdown("---")
    st.subheader(t["res_h"])
    m1, m2, m3 = st.columns(3)
    m1.metric(t["cat"], t["classes"][res_idx])
    m2.metric(t["bmi_lab"], f"{bmi_v:.2f}")
    m3.metric(t["pp_lab"], pp_v)
    
    # الجدول
    plan_df = pd.DataFrame({
        "Day": t["days"], "Workout": t["workouts"], 
        "Nutrition": t["nutrition"], "Advice": t["advice"]
    })
    st.table(plan_df)
    
    # PDF
    pdf_data = create_pdf(t, {t["labels"][0]: age, "BMI": f"{bmi_v:.2f}", t["cat"]: t["classes"][res_idx]}, plan_df)
    st.download_button(t["pdf_btn"], data=pdf_data, file_name=f"Report_{age}.pdf")

st.markdown("---")
st.caption("© 2026 Ahmed Khaled Gamal | All Rights Reserved")

