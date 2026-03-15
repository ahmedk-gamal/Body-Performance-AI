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

# 🎨 Apple Health Style CSS
st.markdown("""
    <style>
    :root { --apple-red: #ff2d55; --apple-bg: #f2f2f7; }
    .main { background-color: var(--apple-bg); }
    
    /* تصميم الأزرار */
    .stButton>button { 
        width: 100%; border-radius: 12px; height: 3.5em; 
        background-color: #ff2d55; color: white; font-weight: 600; 
        font-size: 18px; border: none; transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #e6264d;
        transform: translateY(-2px);
    }

    /* تنسيق السايد بار والروابط */
    .social-links a {
        text-decoration: none;
        color: #1c1c1e;
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px;
        border-radius: 8px;
        transition: 0.3s;
    }
    .social-links a:hover {
        background-color: rgba(255, 45, 85, 0.1);
        color: #ff2d55;
    }

    th { background-color: #ff2d55 !important; color: white !important; }
    [data-testid="stMetricValue"] { color: #ff2d55; font-weight: 800; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 2. دالة توليد التقرير PDF (بما في ذلك السايد بار واللوجو)
def create_pdf(t, data_summary, plan_df):
    pdf = FPDF()
    pdf.add_page()
    
    # رسم خط جانبي أحمر (Sidebar Effect)
    pdf.set_draw_color(255, 45, 85)
    pdf.set_line_width(1)
    pdf.line(65, 10, 65, 280)

    # بيانات المطور (Sidebar in PDF)
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(255, 45, 85)
    pdf.text(10, 25, "HEALTH AI")
    
    pdf.set_font("Arial", 'B', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.text(10, 35, "DEVELOPER:")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 10)
    pdf.text(10, 42, "Ahmed Khaled Gamal")
    
    pdf.set_font("Arial", 'B', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.text(10, 55, "CONTACT:")
    pdf.set_font("Arial", '', 9)
    pdf.set_text_color(0, 0, 255)
    pdf.text(10, 62, "LinkedIn: k-ahmed-auc")
    pdf.text(10, 68, "GitHub: ahmedk-gamal")

    # المحتوى الرئيسي (Main Content)
    pdf.set_xy(70, 20)
    pdf.set_font("Arial", 'B', 22)
    pdf.set_text_color(255, 45, 85)
    pdf.cell(130, 15, txt="HEALTH ANALYSIS REPORT", ln=True, align='L')
    
    pdf.set_xy(70, 45)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(130, 10, txt="1. Summary Results:", ln=True)
    
    pdf.set_font("Arial", '', 11)
    for key, value in data_summary.items():
        pdf.set_x(75)
        pdf.cell(130, 8, txt=f"- {key}: {value}", ln=True)
    
    pdf.ln(10)
    pdf.set_x(70)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(130, 10, txt="2. Weekly Fitness Plan:", ln=True)
    
    # جدول التقرير
    pdf.set_font("Arial", 'B', 10)
    pdf.set_fill_color(255, 45, 85)
    pdf.set_text_color(255, 255, 255)
    pdf.set_x(70)
    pdf.cell(20, 10, "Day", 1, 0, 'C', True)
    pdf.cell(40, 10, "Workout", 1, 0, 'C', True)
    pdf.cell(35, 10, "Nutrition", 1, 0, 'C', True)
    pdf.cell(35, 10, "Advice", 1, 1, 'C', True)
    
    pdf.set_font("Arial", '', 9)
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
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except: return None

# رابط قلب نابض مباشر وسريع جداً
lottie_heart = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_m6cu96.json")

@st.cache_resource
def load_assets():
    try:
        m = load_model('body_performance_model.keras')
        s = joblib.load('scaler.pkl')
        return m, s
    except: return None, None

model, scaler = load_assets()

# 4. السايد بار (مقر الهوية البصرية)
with st.sidebar:
    if lottie_heart:
        st_lottie(lottie_heart, height=180, key="main_heart")
    else:
        st.title("❤️ Health AI")
    
    lang = st.radio("🌐 Language / اللغة", ("English", "العربية"))
    st.markdown("---")
    
    st.subheader("👨‍💻 Developer")
    st.markdown(f"**Ahmed Khaled Gamal**")
    st.markdown(f"""
    <div class="social-links">
        <a href="https://www.linkedin.com/in/k-ahmed-auc/" target="_blank">LinkedIn</a>
        <a href="https://github.com/ahmedk-gamal" target="_blank">GitHub</a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("Technology: Deep Learning (MLP)")

# 5. الواجهة الرئيسية والمدخلات
st.title("🧬 Health AI Analysis")
st.markdown("Predictive Health System inspired by Apple Health.")

if model is None:
    st.error("Missing Model Files! Upload 'body_performance_model.keras' and 'scaler.pkl'.")
    st.stop()

with st.form("main_form"):
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
    
    submit = st.form_submit_button("Analyze & Generate Report ✨")

# 6. التوقع والنتائج
if submit:
    gender_val = 0 if gender == "Male" else 1
    bmi = weight / ((height / 100) ** 2)
    pp = sys - dia
    features = np.array([[age, gender_val, height, weight, fat, sys, dia, grip, situps, bend, jump, bmi, pp]])
    scaled = scaler.transform(features)
    pred = model.predict(scaled)
    idx = np.argmax(pred)
    classes = ['A (Excellent)', 'B (Good)', 'C (Fair)', 'D (Poor)']
    
    st.markdown("---")
    st.header("📊 Results")
    res1, res2, res3 = st.columns(3)
    res1.metric("Class", classes[idx])
    res2.metric("BMI", f"{bmi:.2f}")
    res3.metric("Pulse Pressure", pp)

    # بيانات الخطة
    df_plan = pd.DataFrame({
        "Day": ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"],
        "Workout": ["Push", "Pull", "Legs", "Recovery", "HIIT", "Power", "Rest"],
        "Nutrition": ["High Protein", "High Protein", "Carb Load", "Balanced", "Low Carb", "High Protein", "Cheat"],
        "Advice": ["Hydrate", "Stretch", "Sleep 8h", "Active Recovery", "Focus", "Intensity", "Relax"]
    })
    st.table(df_plan)
    
    # استخراج التقرير
    report_data = {"Age": age, "Gender": gender, "BMI": f"{bmi:.2f}", "Category": classes[idx]}
    pdf_output = create_pdf(None, report_data, df_plan)
    
    st.download_button(
        label="📥 Download Detailed PDF Report",
        data=pdf_output,
        file_name=f"Health_Report_{age}.pdf",
        mime="application/pdf"
    )

st.caption("© 2026 Ahmed Khaled Gamal | Property & Tech Consultant")
