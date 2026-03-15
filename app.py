import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from streamlit_lottie import st_lottie
import requests
from fpdf import FPDF 

# ==========================================
# 1. إعدادات الصفحة (Theme & Config)
# ==========================================
st.set_page_config(
    page_title="Health AI | Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. التنسيق الجمالي (Advanced CSS)
# ==========================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    
    :root { 
        --apple-red: #ff2d55; 
        --soft-blue: #f0f7ff; 
        --deep-blue: #1a237e; 
    }
    
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }
    
    .main-header-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-top: -30px;
    }

    .main-title-text {
        color: var(--deep-blue);
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-top: -10px;
        margin-bottom: 20px;
    }

    .sidebar-header {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-bottom: 20px;
    }
    .logo-text {
        color: var(--apple-red);
        font-size: 28px;
        font-weight: 800;
        letter-spacing: 2px;
        text-align: center;
    }

    .stButton>button { 
        width: 100%; border-radius: 12px; height: 3.8em; 
        background-color: var(--apple-red); color: white; font-weight: 700; 
        border: none; transition: all 0.3s ease; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover { 
        background-color: #e6264d; transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }

    .about-box {
        background-color: var(--soft-blue); padding: 20px; border-radius: 15px;
        border-left: 6px solid #2196F3; font-size: 14px; color: var(--deep-blue); 
        margin-bottom: 25px; line-height: 1.6;
    }

    .social-link {
        display: flex; align-items: center; gap: 12px;
        text-decoration: none; font-weight: 700; margin-bottom: 15px; transition: 0.3s;
    }
    .linkedin { color: #0077b5 !important; }
    .github { color: #333 !important; }
    
    th { background-color: var(--apple-red) !important; color: white !important; text-align: center !important; }
    [data-testid="stMetricValue"] { color: var(--apple-red); font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. الدوال المساعدة (تحديث دالة الـ PDF لدعم التوصيات)
# ==========================================
def create_pdf(summary, plan_df):
    pdf = FPDF()
    pdf.add_page()
    
    # تصميم الخلفية الجانبية
    pdf.set_fill_color(255, 45, 85)
    pdf.rect(0, 0, 65, 300, 'F')
    
    # نصوص الهوية
    pdf.set_font("Arial", 'B', 16); pdf.set_text_color(255, 255, 255)
    pdf.text(10, 30, "HEALTH AI")
    pdf.set_font("Arial", '', 10); pdf.text(10, 50, "DEVELOPER:")
    pdf.set_font("Arial", 'B', 10); pdf.text(10, 56, "Ahmed Khaled Gamal")
    
    # عنوان التقرير
    pdf.set_xy(75, 30); pdf.set_text_color(0, 0, 0); pdf.set_font("Arial", 'B', 22)
    pdf.cell(130, 10, "OFFICIAL HEALTH REPORT", ln=True)
    pdf.ln(10)
    
    # كتابة النتائج الحيوية
    pdf.set_font("Arial", 'B', 14); pdf.cell(130, 10, "BIOMETRIC ANALYSIS", ln=True)
    pdf.set_font("Arial", 'B', 11)
    for k, v in summary.items():
        pdf.set_x(75)
        pdf.cell(50, 8, f"{k}:", 0)
        pdf.set_font("Arial", '', 11)
        pdf.cell(80, 8, f"{str(v)}", ln=True)
        pdf.set_font("Arial", 'B', 11)
    
    pdf.ln(10)
    
    # إضافة جدول التوصيات للـ PDF
    pdf.set_x(75); pdf.set_font("Arial", 'B', 14)
    pdf.cell(130, 10, "YOUR WEEKLY HEALTH PLAN", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Arial", '', 10)
    for index, row in plan_df.iterrows():
        pdf.set_x(75)
        # تنظيف النصوص لضمان التوافق (English only for PDF Stability)
        day = str(row[0]).split(' / ')[0] # نأخذ الجزء الإنجليزي فقط للـ PDF
        work = str(row[1]).split(' / ')[0]
        nutri = str(row[2]).split(' / ')[0]
        
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(130, 6, f"> {day}:", ln=True)
        pdf.set_x(80); pdf.set_font("Arial", '', 10)
        pdf.multi_cell(125, 5, f"Workout: {work} | Nutrition: {nutri}", border=0)
        pdf.ln(2)
        
    return bytes(pdf.output())

def load_lottie(url):
    try: return requests.get(url).json()
    except: return None

@st.cache_resource
def load_ai_models():
    try:
        m = load_model('body_performance_model.keras')
        s = joblib.load('scaler.pkl')
        return m, s
    except: return None, None

heart_anim = load_lottie("https://assets5.lottiefiles.com/packages/lf20_m6cu96.json")
model, scaler = load_ai_models()

# ==========================================
# 4. قاموس اللغات
# ==========================================
texts = {
    "English": {
        "title": "Health AI Dashboard",
        "brief_h": "🤖 Technical Brief",
        "brief_txt": "This system utilizes a <b>Deep Learning MLP</b> model to classify physical performance. It features custom <b>Feature Engineering</b> (BMI & Pulse Pressure).",
        "dev_h": "👨‍💻 Developer Information",
        "btn": "Analyze & Generate Report ✨",
        "pdf_btn": "📥 Download PDF Report",
        "labels": ["Age", "Gender", "Height (cm)", "Weight (kg)", "Body Fat %", "Systolic BP", "Diastolic BP", "Grip Force", "Sit-ups", "Jump (cm)", "Flexibility"],
        "res_h": "📊 Comprehensive Results",
        "cat": "Fitness Category",
        "bmi_lab": "BMI Score",
        "pp_lab": "Pulse Pressure",
        "classes": ['A (Excellent)', 'B (Good)', 'C (Fair)', 'D (Poor)'],
        "days": ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        "workouts": ["Push Day", "Pull Day", "Legs Day", "Recovery", "HIIT", "Power", "Rest"],
        "nutrition": ["High Protein", "High Protein", "Carb Load", "Vitamins", "Low Carb", "Fiber", "Cheat Meal"],
        "advice": ["Hydrate", "Stretch", "8h Sleep", "Walk", "Focus", "Intensity", "Relax"]
    },
    "العربية": {
        "title": "لوحة تحكم الأداء البدني الذكية",
        "brief_h": "🤖 نبذة تقنية",
        "brief_txt": "يعتمد النظام على نموذج <b>التعلم العميق (MLP)</b> للتنبؤ بمستوى اللياقة. تم دمج <b>هندسة البيانات</b> لحساب مؤشر كتلة الجسم وضغط النبض.",
        "dev_h": "👨‍💻 المطور",
        "btn": "تشغيل التحليل واستخراج التقرير ✨",
        "pdf_btn": "📥 تحميل التقرير الرسمي (PDF)",
        "labels": ["العمر", "الجنس", "الطول (سم)", "الوزن (كجم)", "نسبة الدهون %", "الضغط الانقباضي", "الضغط الانبساطي", "قوة القبضة", "تمارين البطن", "الوثب", "المرونة"],
        "res_h": "📊 النتائج والتحليلات",
        "cat": "تصنيف اللياقة",
        "bmi_lab": "مؤشر الكتلة",
        "pp_lab": "ضغط النبض",
        "classes": ['A (ممتاز)', 'B (جيد جداً)', 'C (متوسط)', 'D (ضعيف)'],
        "days": ["السبت", "الأحد", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة"],
        "workouts": ["تمارين دفع", "تمارين سحب", "تمارين أرجل", "استشفاء", "كارديو HIIT", "تمارين قوة", "راحة"],
        "nutrition": ["بروتين عالي", "بروتين عالي", "تحميل كارب", "فيتامينات", "كارب منخفض", "ألياف", "وجبة مفتوحة"],
        "advice": ["شرب ماء", "إطالات", "نوم كافٍ", "مشي", "تركيز", "كثافة", "استشفاء"]
    }
}

# ==========================================
# 5. السايد بار
# ==========================================
with st.sidebar:
    st.markdown('<div class="sidebar-header"><div class="logo-text">HEALTH AI</div></div>', unsafe_allow_html=True)
    lang = st.radio("🌐 Language / اللغة", ("English", "العربية"))
    t = texts[lang]
    st.markdown("---")
    st.subheader(t["brief_h"])
    st.markdown(f'<div class="about-box">{t["brief_txt"]}</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.subheader(t["dev_h"])
    st.markdown(f"**Ahmed Khaled Gamal**")
    st.markdown(f"""
        <a href="https://www.linkedin.com/in/k-ahmed-auc/" class="social-link" target="_blank">LinkedIn</a>
        <a href="https://github.com/ahmedk-gamal" class="social-link" target="_blank">GitHub</a>
    """, unsafe_allow_html=True)

# ==========================================
# 6. المحتوى الرئيسي
# ==========================================
st.markdown('<div class="main-header-container">', unsafe_allow_html=True)
col_l, col_m, col_r = st.columns([1, 1, 1])
with col_m:
    if heart_anim:
        st_lottie(heart_anim, height=200, key="main_heart")
st.markdown(f'<div class="main-title-text">🧬 {t["title"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

if model is None:
    st.error("Model files not found!"); st.stop()

with st.form("health_form"):
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

# ==========================================
# 7. منطق النتائج والتقرير
# ==========================================
if run_btn:
    g_val = 0 if (gender in ["Male", "ذكر"]) else 1
    bmi = w / ((h/100)**2)
    pp = s_bp - d_bp
    
    features = scaler.transform([[age, g_val, h, w, f, s_bp, d_bp, grp, stp, bnd, jmp, bmi, pp]])
    pred_idx = np.argmax(model.predict(features))
    
    st.markdown("---")
    st.subheader(t["res_h"])
    m1, m2, m3 = st.columns(3)
    m1.metric(t["cat"], t["classes"][pred_idx])
    m2.metric(t["bmi_lab"], f"{bmi:.2f}")
    m3.metric(t["pp_lab"], pp)
    
    # بناء جدول التوصيات للواجهة
    plan_df = pd.DataFrame({
        "Day / اليوم": t["days"], 
        "Workout / التمرين": t["workouts"],
        "Nutrition / التغذية": t["nutrition"], 
        "Advice / نصيحة": t["advice"]
    })
    st.table(plan_df)
    
    # بيانات الـ PDF بمفاتيح إنجليزية لضمان الثبات
    pdf_summary = {
        "Age": age,
        "BMI Score": f"{bmi:.2f}",
        "Pulse Pressure": pp,
        "Fitness Level": t["classes"][pred_idx]
    }
    
    # استدعاء الدالة المحدثة التي ترسم الجدول داخل الـ PDF
    pdf_bytes = create_pdf(pdf_summary, plan_df)
    
    st.download_button(
        label=t["pdf_btn"], 
        data=pdf_bytes, 
        file_name="Health_Report.pdf",
        mime="application/pdf"
    )

st.markdown("---")
st.caption("© 2026 Ahmed Khaled Gamal | All Rights Reserved")
