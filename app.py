import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from streamlit_lottie import st_lottie
import requests
from fpdf import FPDF 

# ==============================================================================
# 1. إعدادات الصفحة والهوية البصرية (Theme & Page Configuration)
# ==============================================================================
st.set_page_config(
    page_title="Health AI | Advanced Performance Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. التنسيق الاحترافي (Advanced CSS - Apple Style)
# ==============================================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');
    
    :root { 
        --apple-red: #ff2d55; 
        --soft-blue: #f0f7ff; 
        --deep-blue: #1a237e; 
        --glass-white: rgba(255, 255, 255, 0.8);
    }
    
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }
    .main { background-color: #f5f5f7; }

    /* حاوية العنوان الرئيسي واللوجو الممركز */
    .main-header-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-top: -60px;
        margin-bottom: 30px;
        padding: 20px;
    }
    .main-title {
        color: var(--deep-blue);
        font-weight: 800;
        font-size: 42px;
        margin-top: -15px;
        text-align: center;
        letter-spacing: -1px;
    }

    /* تنسيق السايد بار */
    .sidebar-header-text {
        color: var(--apple-red);
        font-size: 24px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 15px;
        letter-spacing: 1px;
    }

    /* أزرار Apple Style */
    .stButton>button { 
        width: 100%; border-radius: 14px; height: 4em; 
        background-color: var(--apple-red); color: white; font-weight: 700; 
        border: none; transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
        box-shadow: 0 4px 15px rgba(255, 45, 85, 0.3);
    }
    .stButton>button:hover { 
        background-color: #e6264d; transform: translateY(-3px); 
        box-shadow: 0 8px 25px rgba(255, 45, 85, 0.4);
    }

    /* صندوق النبذة التقنية المطور */
    .about-box {
        background-color: var(--soft-blue); padding: 22px; border-radius: 18px;
        border-left: 8px solid #2196F3; font-size: 14.5px; color: var(--deep-blue); 
        line-height: 1.7; box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    /* أيقونات التواصل الاجتماعي الملونة */
    .social-link {
        display: flex; align-items: center; gap: 14px;
        text-decoration: none; font-weight: 700; margin-bottom: 18px; 
        padding: 10px; border-radius: 10px; transition: 0.3s;
    }
    .social-link:hover { background-color: #f0f0f5; transform: translateX(8px); }
    .linkedin-text { color: #0077b5 !important; }
    .github-text { color: #333 !important; }
    
    /* تنسيق الجداول والـ Metrics */
    th { background-color: var(--apple-red) !important; color: white !important; font-size: 16px !important; }
    [data-testid="stMetricValue"] { color: var(--apple-red); font-weight: 800; font-size: 2.2rem; }
    [data-testid="stMetricLabel"] { font-weight: 600; font-size: 1rem; }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. دالة توليد التقرير PDF (دعم كامل للتنسيق الاحترافي)
# ==============================================================================
def create_pdf(t_dict, data_summary, plan_table):
    pdf = FPDF()
    pdf.add_page()
    
    # تصميم الهوية البصرية للتقرير
    pdf.set_fill_color(255, 45, 85)
    pdf.rect(0, 0, 65, 300, 'F')
    
    # بيانات المطور واللوجو (Header)
    pdf.set_font("Arial", 'B', 18); pdf.set_text_color(255, 255, 255)
    pdf.text(12, 35, "HEALTH AI")
    pdf.set_font("Arial", '', 10); pdf.text(12, 55, "DEVELOPED BY:")
    pdf.set_font("Arial", 'B', 11); pdf.text(12, 62, "Ahmed Khaled Gamal")
    
    # العنوان الرئيسي للتقرير
    pdf.set_xy(75, 35); pdf.set_text_color(0, 0, 0); pdf.set_font("Arial", 'B', 24)
    pdf.cell(130, 10, "HEALTH ANALYSIS REPORT", ln=True)
    pdf.set_draw_color(200, 200, 200); pdf.line(75, 50, 200, 50)
    pdf.ln(20)
    
    # محتوى النتائج
    pdf.set_font("Arial", 'B', 14); pdf.set_text_color(255, 45, 85)
    pdf.set_x(75); pdf.cell(100, 10, "1. Biological Metrics Summary:", ln=True)
    pdf.set_font("Arial", '', 12); pdf.set_text_color(50, 50, 50)
    
    for key, value in data_summary.items():
        pdf.set_x(80); pdf.cell(60, 10, f"- {key}:", 0)
        pdf.set_font("Arial", 'B', 12); pdf.cell(60, 10, f"{value}", ln=True)
        pdf.set_font("Arial", '', 12)
        
    return bytes(pdf.output())

# ==============================================================================
# 4. تحميل الأصول والذكاء الاصطناعي (Models & Assets)
# ==============================================================================
def get_lottie_obj(url):
    try: return requests.get(url, timeout=5).json()
    except: return None

heart_lottie = get_lottie_obj("https://assets5.lottiefiles.com/packages/lf20_m6cu96.json")

@st.cache_resource
def load_deep_learning_assets():
    try:
        loaded_model = load_model('body_performance_model.keras')
        loaded_scaler = joblib.load('scaler.pkl')
        return loaded_model, loaded_scaler
    except Exception as e:
        return None, None

model, scaler = load_deep_learning_assets()

# ==============================================================================
# 5. القاموس الضخم للمحتوى (The Massive Multi-language Dictionary)
# ==============================================================================
texts = {
    "English": {
        "main_title": "HEALTH AI PERFORMANCE",
        "brief_h": "🤖 Technical Architecture",
        "brief_txt": "This dashboard is powered by a <b>Deep Learning MLP</b> model. It utilizes <b>Feature Engineering</b> to calculate BMI and Pulse Pressure, providing a comprehensive physical classification.",
        "dev_h": "👨‍💻 Developer Portfolio",
        "btn": "Run AI Diagnostic & Export Report ✨",
        "pdf_btn": "📥 Download Official PDF Report",
        "labels": ["Age", "Gender", "Height (cm)", "Weight (kg)", "Body Fat %", "Systolic BP", "Diastolic BP", "Grip Force", "Sit-ups", "Broad Jump", "Flexibility"],
        "res_h": "📊 Clinical Analysis Results",
        "cat": "Performance Category",
        "bmi_lab": "Body Mass Index (BMI)",
        "pp_lab": "Pulse Pressure (PP)",
        "classes": ['A (Excellent)', 'B (Good)', 'C (Fair)', 'D (Poor)'],
        "days": ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        "workouts": ["Push Power", "Pull Strength", "Legs/Lower", "Active Recovery", "HIIT Session", "Endurance", "Complete Rest"],
        "nutrition": ["High Protein", "High Protein", "Carb Loading", "Micro-nutrients", "Low Carb", "Lean Protein", "Cheat Meal"],
        "advice": ["Hydrate 3L", "Dynamic Stretch", "Deep Sleep", "Nature Walk", "Mental Focus", "High Intensity", "Mindfulness"]
    },
    "العربية": {
        "main_title": "لوحة تحكم الصحة والذكاء الاصطناعي",
        "brief_h": "🤖 البنية التقنية للمشروع",
        "brief_txt": "يعتمد هذا النظام على نموذج <b>التعلم العميق (MLP)</b> لتصنيف الأداء البدني. تم دمج <b>هندسة البيانات</b> لحساب مؤشر كتلة الجسم وضغط النبض بشكل آلي.",
        "dev_h": "👨‍💻 معلومات المطور",
        "btn": "بدء التحليل واستخراج التقرير ✨",
        "pdf_btn": "📥 تحميل التقرير الطبي الرسمي (PDF)",
        "labels": ["العمر", "الجنس", "الطول (سم)", "الوزن (كجم)", "نسبة الدهون %", "الضغط الانقباضي", "الضغط الانبساطي", "قوة القبضة", "تمارين البطن", "الوثب العريض", "المرونة"],
        "res_h": "📊 نتائج التحليل السريري",
        "cat": "فئة الأداء البدني",
        "bmi_lab": "مؤشر كتلة الجسم",
        "pp_lab": "ضغط النبض (PP)",
        "classes": ['A (ممتاز)', 'B (جيد جداً)', 'C (متوسط)', 'D (ضعيف)'],
        "days": ["السبت", "الأحد", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة"],
        "workouts": ["تمارين الدفع", "تمارين السحب", "تمارين الأرجل", "استشفاء نشط", "كارديو HIIT", "تمارين التحمل", "راحة تامة"],
        "nutrition": ["بروتين عالي", "بروتين عالي", "تحميل كربوهيدرات", "فيتامينات", "كربوهيدرات منخفضة", "بروتين وألياف", "وجبة مفتوحة"],
        "advice": ["شرب 3 لتر ماء", "إطالات حركية", "نوم عميق", "مشي في الطبيعة", "تركيز ذهني", "كثافة عالية", "تأمل واستجمام"]
    }
}

# ==============================================================================
# 6. بناء الواجهة (The Master UI Construction)
# ==============================================================================

# --- السايد بار (Sidebar) ---
with st.sidebar:
    st.markdown('<div class="sidebar-header-text">HEALTH AI</div>', unsafe_allow_html=True)
    selected_lang = st.radio("🌐 Language / اللغة", ("English", "العربية"), label_visibility="collapsed")
    t = texts[selected_lang]
    
    st.markdown("---")
    st.subheader(t["brief_h"])
    st.markdown(f'<div class="about-box">{t["brief_txt"]}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader(t["dev_h"])
    st.markdown(f"**Ahmed Khaled Gamal**\n\nProperty & Tech Consultant")
    
    # روابط التواصل الاجتماعي بالأيقونات
    st.markdown(f"""
        <a href="https://www.linkedin.com/in/k-ahmed-auc/" class="social-link" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25"> 
            <span class="linkedin-text">LinkedIn Profile</span>
        </a>
        <a href="https://github.com/ahmedk-gamal" class="social-link" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="25"> 
            <span class="github-text">GitHub Repository</span>
        </a>
    """, unsafe_allow_html=True)

# --- الواجهة الرئيسية (Main Body) ---
st.markdown('<div class="main-header-container">', unsafe_allow_html=True)
if heart_lottie:
    st_lottie(heart_lottie, height=200, key="main_heart_anim")
st.markdown(f'<div class="main-title">{t["main_title"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

if model is None:
    st.error("Critical Error: Model files not found in the directory!"); st.stop()

# نموذج إدخال البيانات
with st.form("physical_analysis_form"):
    col1, col2 = st.columns(2)
    with col1:
        u_age = st.number_input(t["labels"][0], 10, 90, 25)
        u_gen = st.selectbox(t["labels"][1], ["Male", "Female"] if selected_lang == "English" else ["ذكر", "أنثى"])
        u_height = st.number_input(t["labels"][2], 120.0, 220.0, 175.0)
        u_weight = st.number_input(t["labels"][3], 30.0, 200.0, 75.0)
        u_fat = st.number_input(t["labels"][4], 5.0, 50.0, 22.0)
    with col2:
        u_sys = st.number_input(t["labels"][5], 80, 200, 120)
        u_dia = st.number_input(t["labels"][6], 50, 120, 80)
        u_grip = st.number_input(t["labels"][7], 10.0, 100.0, 45.0)
        u_situps = st.number_input(t["labels"][8], 0, 100, 35)
        u_jump = st.number_input(t["labels"][9], 50.0, 350.0, 215.0)
        u_bend = st.number_input(t["labels"][10], -25.0, 55.0, 15.0)
    
    submit_analysis = st.form_submit_button(t["btn"])

# ==============================================================================
# 7. معالجة النتائج وتوليد المخرجات (Analysis & Output)
# ==============================================================================
if submit_analysis:
    # حساب هندسة البيانات (Feature Engineering)
    gender_binary = 0 if (u_gen in ["Male", "ذكر"]) else 1
    calculated_bmi = u_weight / ((u_height/100)**2)
    calculated_pp = u_sys - u_dia
    
    # تحضير المصفوفة وتوقع النتيجة
    input_vector = np.array([[u_age, gender_binary, u_height, u_weight, u_fat, u_sys, u_dia, u_grip, u_situps, u_bend, u_jump, calculated_bmi, calculated_pp]])
    scaled_inputs = scaler.transform(input_vector)
    prediction_array = model.predict(scaled_inputs)
    final_class_idx = np.argmax(prediction_array)
    
    # عرض النتائج في بطاقات (Metrics)
    st.markdown("---")
    st.subheader(t["res_h"])
    m1, m2, m3 = st.columns(3)
    m1.metric(t["cat"], t["classes"][final_class_idx])
    m2.metric(t["bmi_lab"], f"{calculated_bmi:.2f}")
    m3.metric(t["pp_lab"], f"{calculated_pp}")
    
    # جدول الخطة الأسبوعية
    st.markdown("<br>", unsafe_allow_html=True)
    plan_data = {
        "Day / اليوم": t["days"],
        "Workout / التمرين": t["workouts"],
        "Nutrition / التغذية": t["nutrition"],
        "Pro Advice / نصيحة": t["advice"]
    }
    df_plan = pd.DataFrame(plan_data)
    st.table(df_plan)
    
    # توليد وتحميل ملف التقرير PDF
    summary_results = {
        t["labels"][0]: u_age,
        "BMI": f"{calculated_bmi:.2f}",
        "Pulse Pressure": calculated_pp,
        t["cat"]: t["classes"][final_class_idx]
    }
    pdf_report = create_pdf(t, summary_results, df_plan)
    st.download_button(t["pdf_btn"], data=pdf_report, file_name=f"Health_Report_{u_age}.pdf", mime="application/pdf")

# تذييل الصفحة (Footer)
st.markdown("---")
st.caption("© 2026 Ahmed Khaled Gamal | Property Consultant & AI Developer | Cairo, Egypt")
# ==============================================================================
# End of Code - أسطر الكود كاملة بدون أي اختصار
# ==============================================================================
