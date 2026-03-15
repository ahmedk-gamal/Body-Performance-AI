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

# 🎨 CSS المحسن (تعديل ألوان النبذة والروابط)
st.markdown("""
    <style>
    :root { --apple-red: #ff2d55; --soft-blue: #e3f2fd; --deep-blue: #1565c0; }
    .main { background-color: #f2f2f7; }
    
    /* تصميم الأزرار */
    .stButton>button { 
        width: 100%; border-radius: 12px; height: 3.5em; 
        background-color: var(--apple-red); color: white; font-weight: 600; 
        font-size: 18px; border: none; transition: 0.3s;
    }
    .stButton>button:hover { background-color: #e6264d; transform: translateY(-2px); }

    /* ألوان الروابط الأصلية */
    .linkedin-link { color: #0077b5 !important; font-weight: bold; text-decoration: none; display: flex; align-items: center; gap: 10px; }
    .github-link { color: #333 !important; font-weight: bold; text-decoration: none; display: flex; align-items: center; gap: 10px; }
    
    /* صندوق النبذة التقنية (الألوان الجديدة) */
    .about-box {
        background-color: var(--soft-blue);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid var(--deep-blue);
        font-size: 14px;
        color: #0d47a1; /* نص داكن وواضح */
        margin-bottom: 20px;
    }
    
    th { background-color: var(--apple-red) !important; color: white !important; }
    [data-testid="stMetricValue"] { color: var(--apple-red); font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

# 2. دالة توليد التقرير PDF (مع بيانات المطور)
def create_pdf(data_summary, plan_df):
    pdf = FPDF()
    pdf.add_page()
    
    # Sidebar design in PDF
    pdf.set_draw_color(255, 45, 85); pdf.set_line_width(1); pdf.line(65, 10, 65, 280)

    # Info Side
    pdf.set_font("Arial", 'B', 14); pdf.set_text_color(255, 45, 85); pdf.text(10, 25, "HEALTH AI")
    pdf.set_font("Arial", 'B', 10); pdf.set_text_color(0, 0, 0); pdf.text(10, 40, "DEVELOPER:")
    pdf.set_font("Arial", '', 10); pdf.text(10, 47, "Ahmed Khaled Gamal")
    
    # Main Side
    pdf.set_xy(70, 20); pdf.set_font("Arial", 'B', 20); pdf.set_text_color(255, 45, 85)
    pdf.cell(130, 10, "HEALTH ANALYSIS REPORT", ln=True)
    pdf.ln(10); pdf.set_font("Arial", '', 11); pdf.set_text_color(0, 0, 0)
    for key, value in data_summary.items():
        pdf.set_x(70); pdf.cell(130, 8, txt=f"- {key}: {value}", ln=True)
    
    # Table headers
    pdf.ln(10); pdf.set_font("Arial", 'B', 10); pdf.set_fill_color(255, 45, 85); pdf.set_text_color(255, 255, 255)
    pdf.set_x(70); pdf.cell(20, 10, "Day", 1, 0, 'C', True); pdf.cell(40, 10, "Workout", 1, 0, 'C', True)
    pdf.cell(35, 10, "Nutrition", 1, 0, 'C', True); pdf.cell(35, 10, "Advice", 1, 1, 'C', True)
    
    # Table rows
    pdf.set_font("Arial", '', 9); pdf.set_text_color(0, 0, 0)
    for index, row in plan_df.iterrows():
        pdf.set_x(70); pdf.cell(20, 10, str(row[0]), 1); pdf.cell(40, 10, str(row[1]), 1)
        pdf.cell(35, 10, str(row[2]), 1); pdf.cell(35, 10, str(row[3]), 1); pdf.ln()
        
    return bytes(pdf.output())

# تابع للجزء الأول.. الصقه في نهاية الملف

# 3. تحميل الأصول
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

# 4. السايد بار (مقر الهوية البصرية)
with st.sidebar:
    if lottie_heart:
        st_lottie(lottie_heart, height=160, key="heart")
    else:
        st.title("❤️ Health AI")
    
    st.markdown("---")
    st.subheader("🤖 Technical Brief")
    st.markdown("""
    <div class="about-box">
    Built with a <b>Multi-Layer Perceptron (MLP)</b>, this AI predicts physical class based on biological metrics. 
    It features custom <b>Feature Engineering</b> (BMI & Pulse Pressure) to enhance accuracy.
    </div>
    """, unsafe_allow_html=True)
    
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
    lang = st.radio("🌐 Language", ("English", "العربية"))

# 5. الواجهة الرئيسية واستمارة البيانات
st.title("🧬 Health AI Analysis")
if model is None:
    st.error("Model files missing!"); st.stop()

with st.form("main_form"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 10, 90, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height (cm)", 120.0, 220.0, 175.0)
        weight = st.number_input("Weight (kg)", 30.0, 200.0, 75.0)
        fat = st.number_input("Fat %", 5.0, 50.0, 20.0)
    with c2:
        sys = st.number_input("Systolic BP", 80, 200, 120); dia = st.number_input("Diastolic BP", 50, 120, 80)
        grip = st.number_input("Grip Force", 10.0, 100.0, 45.0); situps = st.number_input("Sit-ups", 0, 100, 35)
        jump = st.number_input("Jump (cm)", 50.0, 350.0, 210.0); bend = st.number_input("Flexibility", -20.0, 50.0, 15.0)
    submit = st.form_submit_button("Analyze & Generate Report ✨")

if submit:
    gender_val = 0 if gender == "Male" else 1
    bmi = weight / ((height/100)**2); pp = sys - dia
    features = np.array([[age, gender_val, height, weight, fat, sys, dia, grip, situps, bend, jump, bmi, pp]])
    scaled = scaler.transform(features)
    pred = model.predict(scaled); idx = np.argmax(pred)
    classes = ['A (Excellent)', 'B (Good)', 'C (Fair)', 'D (Poor)']
    
    st.markdown("---")
    r1, r2, r3 = st.columns(3)
    r1.metric("Category", classes[idx]); r2.metric("BMI", f"{bmi:.2f}"); r3.metric("Pulse Pressure", pp)

    df_plan = pd.DataFrame({
        "Day": ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"],
        "Workout": ["Push", "Pull", "Legs", "Recovery", "HIIT", "Power", "Rest"],
        "Nutrition": ["Clean", "Clean", "Carb Load", "Vitamins", "Protein", "Protein", "Cheat"],
        "Advice": ["Hydrate", "Stretch", "Sleep", "Active", "Intensity", "Focus", "Relax"]
    })
    st.table(df_plan)
    
    pdf_bytes = create_pdf({"Age": age, "BMI": f"{bmi:.2f}", "Category": classes[idx]}, df_plan)
    st.download_button("📥 Download PDF Report", data=pdf_bytes, file_name="Health_Report.pdf")

st.caption("© 2026 Ahmed Khaled Gamal | All Rights Reserved")
