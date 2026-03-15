import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from streamlit_lottie import st_lottie
import requests
from fpdf import FPDF
import os

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
# 3. الدوال المساعدة - FIX: Unicode PDF
# ==========================================

def download_dejavu_font():
    """تحميل خط DejaVu يدعم Unicode كامل"""
    font_path = "DejaVuSans.ttf"
    font_bold_path = "DejaVuSans-Bold.ttf"
    
    if not os.path.exists(font_path):
        url = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf"
        try:
            r = requests.get(url, timeout=10)
            with open(font_path, 'wb') as f:
                f.write(r.content)
        except:
            return None, None
            
    if not os.path.exists(font_bold_path):
        url = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans-Bold.ttf"
        try:
            r = requests.get(url, timeout=10)
            with open(font_bold_path, 'wb') as f:
                f.write(r.content)
        except:
            return font_path, None
            
    return font_path, font_bold_path


def create_pdf(summary, plan_df=None):
    font_path, font_bold_path = download_dejavu_font()
    
    # استخدام FPDF2 مع Unicode
    pdf = FPDF()
    pdf.add_page()
    
    # إضافة خط Unicode
    if font_path and os.path.exists(font_path):
        pdf.add_font("DejaVu", "", font_path, uni=True)
        if font_bold_path and os.path.exists(font_bold_path):
            pdf.add_font("DejaVu", "B", font_bold_path, uni=True)
            bold_font = "DejaVu"
        else:
            bold_font = "DejaVu"
        main_font = "DejaVu"
    else:
        # Fallback: تنظيف النص من Unicode غير المدعوم
        main_font = "Arial"
        bold_font = "Arial"

    def safe_text(text):
        """تنظيف النص إذا استخدمنا Arial fallback"""
        if main_font == "Arial":
            return str(text).encode('latin-1', 'replace').decode('latin-1')
        return str(text)

    # ===== تصميم الـ PDF =====
    
    # الشريط الجانبي الأحمر
    pdf.set_fill_color(255, 45, 85)
    pdf.rect(0, 0, 65, 300, 'F')
    
    # نصوص الهوية
    pdf.set_font(bold_font, 'B', 16)
    pdf.set_text_color(255, 255, 255)
    pdf.text(8, 25, "HEALTH AI")
    
    pdf.set_font(main_font, '', 9)
    pdf.text(8, 38, "DEVELOPER:")
    pdf.set_font(bold_font, 'B', 9)
    pdf.text(8, 45, "Ahmed Khaled")
    pdf.text(8, 52, "Gamal")
    
    pdf.set_font(main_font, '', 8)
    pdf.text(8, 65, "linkedin.com/in/")
    pdf.text(8, 71, "k-ahmed-auc/")
    pdf.text(8, 82, "github.com/")
    pdf.text(8, 88, "ahmedk-gamal")

    # خط فاصل
    pdf.set_draw_color(255, 255, 255)
    pdf.set_line_width(0.5)
    pdf.line(8, 97, 57, 97)

    pdf.set_font(main_font, '', 8)
    pdf.set_text_color(255, 200, 200)
    pdf.text(8, 105, "© 2026 Health AI")
    pdf.text(8, 111, "All Rights Reserved")

    # ===== المحتوى الرئيسي =====
    
    # العنوان الرئيسي
    pdf.set_xy(75, 15)
    pdf.set_text_color(30, 30, 30)
    pdf.set_font(bold_font, 'B', 20)
    pdf.cell(130, 10, safe_text("OFFICIAL HEALTH REPORT"), ln=True)
    
    pdf.set_xy(75, 27)
    pdf.set_font(main_font, '', 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(130, 8, safe_text("Generated by Health AI Dashboard"), ln=True)
    
    # خط فاصل
    pdf.set_draw_color(255, 45, 85)
    pdf.set_line_width(1)
    pdf.line(75, 38, 200, 38)
    pdf.ln(5)

    # ===== قسم النتائج =====
    pdf.set_xy(75, 45)
    pdf.set_font(bold_font, 'B', 13)
    pdf.set_text_color(26, 35, 126)
    pdf.cell(130, 8, safe_text("Health Metrics"), ln=True)
    pdf.ln(2)

    row_colors = [(240, 247, 255), (255, 255, 255)]
    for i, (k, v) in enumerate(summary.items()):
        pdf.set_x(75)
        pdf.set_fill_color(*row_colors[i % 2])
        pdf.set_text_color(50, 50, 50)
        pdf.set_font(bold_font, 'B', 11)
        pdf.cell(70, 9, safe_text(str(k)), fill=True)
        pdf.set_font(main_font, '', 11)
        pdf.set_text_color(255, 45, 85)
        pdf.cell(60, 9, safe_text(str(v)), fill=True, ln=True)

    pdf.ln(8)

    # ===== خطة الأسبوع =====
    if plan_df is not None:
        pdf.set_x(75)
        pdf.set_font(bold_font, 'B', 13)
        pdf.set_text_color(26, 35, 126)
        pdf.cell(130, 8, safe_text("Weekly Plan"), ln=True)
        pdf.ln(2)

        # رأس الجدول
        headers = ["Day", "Workout", "Nutrition", "Advice"]
        col_widths = [25, 40, 35, 30]
        
        pdf.set_x(75)
        pdf.set_fill_color(255, 45, 85)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font(bold_font, 'B', 9)
        for h_text, w in zip(headers, col_widths):
            pdf.cell(w, 8, safe_text(h_text), border=1, fill=True, align='C')
        pdf.ln()

        # بيانات الجدول
        for i, row in plan_df.iterrows():
            pdf.set_x(75)
            pdf.set_fill_color(*row_colors[i % 2])
            pdf.set_text_color(50, 50, 50)
            pdf.set_font(main_font, '', 8)
            row_values = list(row.values)
            for val, w in zip(row_values, col_widths):
                # تنظيف قيم العربية للجدول
                cell_text = safe_text(str(val))
                pdf.cell(w, 7, cell_text, border=1, fill=True, align='C')
            pdf.ln()

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
        "days": ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"],
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
        <a href="https://www.linkedin.com/in/k-ahmed-auc/" class="social-link linkedin" target="_blank">LinkedIn</a>
        <a href="https://github.com/ahmedk-gamal" class="social-link github" target="_blank">GitHub</a>
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
    
    plan_df = pd.DataFrame({
        "Day / اليوم": t["days"], "Workout / التمرين": t["workouts"],
        "Nutrition / التغذية": t["nutrition"], "Advice / نصيحة": t["advice"]
    })
    st.table(plan_df)
    
    # بيانات الـ PDF بمفاتيح إنجليزية
    pdf_summary = {
        "Age": age,
        "BMI Score": f"{bmi:.2f}",
        "Pulse Pressure": pp,
        "Fitness Category": t["classes"][pred_idx]
    }
    
    # إنشاء الـ PDF مع تمرير الجدول
    pdf_bytes = create_pdf(pdf_summary, plan_df)
    
    st.download_button(
        label=t["pdf_btn"], 
        data=pdf_bytes, 
        file_name="Health_Report.pdf",
        mime="application/pdf"
    )

st.markdown("---")
st.caption("© 2026 Ahmed Khaled Gamal | All Rights Reserved")
