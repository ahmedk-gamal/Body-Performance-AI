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
    
    /* تنسيق الهيدر في السايد بار */
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
        margin-top: -15px;
        margin-bottom: 10px;
    }

    /* أزرار أبل */
    .stButton>button { 
        width: 100%; border-radius: 12px; height: 3.8em; 
        background-color: var(--apple-red); color: white; font-weight: 700; 
        border: none; transition: all 0.3s ease; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover { 
        background-color: #e6264d; transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }

    /* صندوق النبذة التقنية الملون */
    .about-box {
        background-color: var(--soft-blue); padding: 20px; border-radius: 15px;
        border-left: 6px solid #2196F3; font-size: 14px; color: var(--deep-blue); 
        margin-bottom: 25px; line-height: 1.6;
    }

    /* أيقونات التواصل الاجتماعي */
    .social-link {
        display: flex; align-items: center; gap: 12px;
        text-decoration: none; font-weight: 700; margin-bottom: 15px; transition: 0.3s;
    }
    .linkedin { color: #0077b5 !important; }
    .github { color: #333 !important; }
    .social-link:hover { transform: translateX(5px); opacity: 0.8; }
    
    /* تنسيق الجداول والنتائج */
    th { background-color: var(--apple-red) !important; color: white !important; text-align: center !important; }
    [data-testid="stMetricValue"] { color: var(--apple-red); font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. الدوال المساعدة (PDF & Assets)
# ==========================================
def create_pdf(t, summary, plan):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(255, 45, 85)
    pdf.rect(0, 0, 65, 300, 'F')
    
    pdf.set_font("Arial", 'B', 16); pdf.set_text_color(255, 255, 255)
    pdf.text(10, 30, "HEALTH AI")
    pdf.set_font("Arial", '', 10); pdf.text(10, 50, "DEVELOPER:")
    pdf.set_font("Arial", 'B', 10); pdf.text(10, 56, "Ahmed Khaled Gamal")
    
    pdf.set_xy(75, 30); pdf.set_text_color(0, 0, 0); pdf.set_font("Arial", 'B', 22)
    pdf.cell(130, 10, "OFFICIAL HEALTH REPORT", ln=True)
    pdf.ln(15)
    
    pdf.set_font("Arial", 'B', 12)
    for k, v in summary.items():
        pdf.set_x(75); pdf.cell(50, 10, f"{k}:", 0)
        pdf.set_font("Arial", '', 12); pdf.cell(80, 10, f"{v}", ln=True)
        pdf.set_font("Arial", 'B', 12)
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

# تحميل الموديل والأنيميشن
heart_anim = load_lottie("https://assets5.lottiefiles.com/packages/lf20_m6cu96.json")
model, scaler = load_ai_models()

# ==========================================
# 4. قاموس اللغات الاحترافي
# ==========================================
texts = {
    "English": {
        "title": "🧬 Health Performance Dashboard",
        "brief_h": "🤖 Technical Brief",
        "brief_txt": "This system utilizes a <b>Deep Learning MLP</b> model to classify physical performance. It features custom <b>Feature Engineering</b> (BMI & Pulse Pressure) to enhance decision-making accuracy.",
        "dev_h": "👨‍💻 Developer Information",
        "btn": "Analyze & Generate Report ✨",
        "pdf_btn": "📥 Download PDF Report",
        "labels": ["Age", "Gender", "Height (cm)", "Weight (kg)", "Body Fat %", "Systolic BP", "Diastolic BP", "Grip Force", "Sit-ups", "Jump (cm)", "Flexibility"],
        "res_h": "📊 Comprehensive Results",
        "cat": "Fitness Category",
        "bmi_lab": "BMI Score",
        "pp_lab": "Pulse Pressure",
        "classes": ['A (Excellent)', 'B (Good)', 'C (Fair)', 'D (Poor)'],
        "days": ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"],
        "workouts": ["Push Day", "Pull Day", "Legs Day", "Recovery", "HIIT", "Power", "Rest"],
        "nutrition": ["High Protein", "High Protein", "Carb Load", "Vitamins", "Low Carb", "Fiber", "Cheat Meal"],
        "advice": ["Hydrate", "Stretch", "8h Sleep", "Walk", "Focus", "Intensity", "Relax"]
    },
    "العربية": {
        "title": "🧬 لوحة تحكم الأداء البدني الذكية",
        "brief_h": "🤖 نبذة تقنية",
        "brief_txt": "يعتمد النظام على نموذج <b>التعلم العميق (MLP)</b> للتنبؤ بمستوى اللياقة. تم دمج <b>هندسة البيانات</b> لحساب مؤشر كتلة الجسم وضغط النبض بدقة.",
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
# 5. السايد بار (The Master Sidebar)
# ==========================================
with st.sidebar:
    # اللوجو وكلمة HEALTH AI
    st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
    if heart_anim:
        st_lottie(heart_anim, height=130, key="heart")
    st.markdown('<div class="logo-text">HEALTH AI</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    lang = st.radio("🌐 Language / اللغة", ("English", "العربية"))
    t = texts[lang]
    
    st.markdown("---")
    st.subheader(t["brief_h"])
    st.markdown(f'<div class="about-box">{t["brief_txt"]}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader(t["dev_h"])
    st.markdown(f"**Ahmed Khaled Gamal**")
    
    # الأيقونات الأصلية
    st.markdown(f"""
        <a href="https://www.linkedin.com/in/k-ahmed-auc/" class="social-link linkedin" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25"> LinkedIn
        </a>
        <a href="https://github.com/ahmedk-gamal" class="social-link github" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="25"> GitHub
        </a>
    """, unsafe_allow_html=True)

# ==========================================
# 6. المحتوى الرئيسي (Main UI)
# ==========================================
st.title(t["title"])

if model is None:
    st.error("Model files (keras/pkl) not found!"); st.stop()

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
# 7. منطق النتائج (Results Logic)
# ==========================================
if run_btn:
    # معالجة البيانات
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
    
    # الجدول المترجم
    plan_df = pd.DataFrame({
        "Day / اليوم": t["days"], "Workout / التمرين": t["workouts"],
        "Nutrition / التغذية": t["nutrition"], "Advice / نصيحة": t["advice"]
    })
    st.table(plan_df)
    
    # التقرير
    sum_data = {t["labels"][0]: age, "BMI": f"{bmi:.2f}", t["cat"]: t["classes"][pred_idx]}
    pdf_bytes = create_pdf(t, sum_data, plan_df)
    st.download_button(t["pdf_btn"], data=pdf_bytes, file_name="Health_Report.pdf")

# Footer
st.markdown("---")
st.caption("© 2026 Ahmed Khaled Gamal | All Rights Reserved")
