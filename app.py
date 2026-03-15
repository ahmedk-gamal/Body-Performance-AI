import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from streamlit_lottie import st_lottie
import requests

# 1. إعدادات الصفحة بنمط Apple
st.set_page_config(
    page_title="Health AI | الأداء البدني",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 Apple Health Style CSS
st.markdown("""
    <style>
    :root {
        --apple-red: #ff2d55;
    }
    .main { background-color: #f2f2f7; }
    
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

    /* تحسين شكل الروابط في السايد بار */
    .social-links a {
        text-decoration: none;
        color: inherit;
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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 2. تحميل انيميشن القلب
def load_lottieurl(url):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except: return None

lottie_heart = load_lottieurl("https://lottie.host/82334807-6c2e-4054-8e36-7c0a6b72a6b2/v8Y8yO5x20.json")

# 3. نظام الترجمة
with st.sidebar:
    if lottie_heart:
        st_lottie(lottie_heart, height=150, key="apple_heart")
    
    lang = st.radio("🌐 Select Language / اختر اللغة", ("English", "العربية"))
    st.markdown("---")

texts = {
    "English": {
        "title": "🧬 Physical Health Analysis",
        "sub": "AI-powered performance tracking inspired by Apple Health.",
        "about_h": "🤖 Technology",
        "model_desc": "**MLP Neural Network**\n3 Hidden Layers (512-256-128).",
        "model_stats": "**Accuracy:** 82.19%\n**Dataset:** 13K+ records",
        "dev_h": "👨‍💻 Developer",
        "basic_h": "📋 Vital Signs",
        "age": "Age", "gender": "Gender", "male": "Male", "female": "Female",
        "height": "Height (cm)", "weight": "Weight (kg)", "fat": "Fat %",
        "test_h": "💪 Activity Tests",
        "sys": "Systolic BP", "dia": "Diastolic BP", "grip": "Grip Force",
        "situps": "Sit-ups", "jump": "Jump (cm)", "bend": "Flexibility",
        "btn": "Calculate My Health ✨",
        "res_h": "📊 Health Summary",
        "class": "Fitness Category", "bmi": "BMI", "pp": "Pulse Pressure",
        "plan_h": "🗓️ Personalized Action Plan",
        "col_day": "Day", "col_exe": "Activity", "col_nut": "Nutrition", "col_tip": "Advice",
        "days": ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"],
        "classes": ['A (Excellent)', 'B (Good)', 'C (Fair)', 'D (Poor)']
    },
    "العربية": {
        "title": "🧬 تحليل المؤشرات الصحية",
        "sub": "تحليل ذكي للأداء البدني مستوحى من Apple Health.",
        "about_h": "🤖 التقنية المستخدمة",
        "model_desc": "**الشبكة العصبية MLP**\n3 طبقات خفية (512-256-128 نيوترون).",
        "model_stats": "**دقة النموذج:** 82.19%\n**البيانات:** 13 ألف+ سجل",
        "dev_h": "👨‍💻 المطور",
        "basic_h": "📋 المؤشرات الحيوية",
        "age": "العمر", "gender": "النوع", "male": "ذكر", "female": "أنثى",
        "height": "الطول (سم)", "weight": "الوزن (كجم)", "fat": "نسبة الدهون %",
        "test_h": "💪 اختبارات النشاط",
        "sys": "الضغط الانقباضي", "dia": "الضغط الانبساطي", "grip": "قوة القبضة",
        "situps": "تمارين البطن", "jump": "القفز (سم)", "bend": "المرونة",
        "btn": "احسب مؤشراتي الآن ✨",
        "res_h": "📊 ملخص الحالة",
        "class": "الفئة البدنية", "bmi": "كتلة الجسم", "pp": "فرق الضغط",
        "plan_h": "🗓️ خطة العمل المقترحة",
        "col_day": "اليوم", "col_exe": "النشاط", "col_nut": "التغذية", "col_tip": "نصيحة",
        "days": ["السبت", "الأحد", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة"],
        "classes": ['A (ممتاز)', 'B (جيد جداً)', 'C (جيد)', 'D (ضعيف)']
    }
}
t = texts[lang]

# 4. عرض بيانات المطور وروابط التواصل في السايد بار
with st.sidebar:
    st.subheader(t["about_h"])
    st.info(t["model_desc"])
    st.caption(t["model_stats"])
    st.markdown("---")
    st.subheader(t["dev_h"])
    st.markdown(f"**Ahmed Khaled Gamal**")
    
    # روابط التواصل الاجتماعي
    st.markdown(f"""
    <div class="social-links">
        <a href="https://www.linkedin.com/in/k-ahmed-auc/" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="20"> LinkedIn
        </a>
        <a href="https://github.com/ahmedk-gamal" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="20"> GitHub
        </a>
    </div>
    """, unsafe_allow_html=True)

# 5. تحميل الموديل والـ Scaler
@st.cache_resource
def load_assets():
    try:
        m = load_model('body_performance_model.keras')
        s = joblib.load('scaler.pkl')
        return m, s
    except: return None, None

model, scaler = load_assets()

if model is None:
    st.error("Missing Model Files! Please ensure 'body_performance_model.keras' and 'scaler.pkl' are uploaded.")
    st.stop()

# 6. الواجهة الرئيسية
st.title(t["title"])
st.markdown(t["sub"])

with st.form("apple_form"):
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(t["basic_h"])
        age = st.number_input(t["age"], 10, 90, 25)
        gender = st.selectbox(t["gender"], [t["male"], t["female"]])
        height = st.number_input(t["height"], 120.0, 220.0, 175.0)
        weight = st.number_input(t["weight"], 30.0, 200.0, 75.0)
        fat = st.number_input(t["fat"], 5.0, 50.0, 20.0)
    with c2:
        st.subheader(t["test_h"])
        sys = st.number_input(t["sys"], 80, 200, 120)
        dia = st.number_input(t["dia"], 50, 120, 80)
        grip = st.number_input(t["grip"], 10.0, 100.0, 45.0)
        situps = st.number_input(t["situps"], 0, 100, 35)
        jump = st.number_input(t["jump"], 50.0, 350.0, 210.0)
        bend = st.number_input(t["bend"], -20.0, 50.0, 15.0)
    
    submit = st.form_submit_button(t["btn"])

if submit:
    gender_val = 0 if gender == t["male"] else 1
    bmi = weight / ((height / 100) ** 2)
    pp = sys - dia
    features = np.array([[age, gender_val, height, weight, fat, sys, dia, grip, situps, bend, jump, bmi, pp]])
    scaled = scaler.transform(features)
    pred = model.predict(scaled)
    idx = np.argmax(pred)
    
    st.markdown("---")
    st.header(t["res_h"])
    r1, r2, r3 = st.columns(3)
    r1.metric(t["class"], t["classes"][idx])
    r2.metric(t["bmi"], f"{bmi:.2f}")
    r3.metric(t["pp"], pp)

    # جداول الخطة
    if lang == "English":
        workouts = {0:["Push Day","Pull Day","Legs","Active Recovery","HIIT","Full Body","Rest"],
                    1:["Chest/Tri","Back/Bi","Rest","Lower Body","Shoulders","Cardio","Rest"],
                    2:["Power Walk","Rest","Bodyweight","Power Walk","Rest","Circuit","Active"],
                    3:["Light Walk","Stretch","Rest","Light Walk","Yoga","Light Walk","Rest"]}
        foods = ["Berries","Broccoli","Avocado","Kiwi","Carrots","Pomegranate","Mix"]
        tips = ["High Carb","Protein","Hydration","Complex Carbs","Antioxidants","Muscle Repair","Relax"]
    else:
        workouts = {0:["دفع","سحب","أرجل","استشفاء","HIIT","قوة شاملة","راحة"],
                    1:["صدر وتراي","ظهر وباي","راحة","أرجل","أكتاف","جري","راحة"],
                    2:["مشي سريع","راحة","وزن جسم","مشي سريع","راحة","دائرة","يوم نشط"],
                    3:["مشي خفيف","إطالات","راحة","مشي خفيف","يوجا","مشي خفيف","راحة"]}
        foods = ["توت","بروكلي","أفوكادو","كيوي","جزر","رمان","مشكل"]
        tips = ["كارب عالي","بروتين","ترطيب","كارب معقد","مضادات أكسدة","ترميم","استرخاء"]

    df = pd.DataFrame({t["col_day"]: t["days"], t["col_exe"]: workouts[idx], t["col_nut"]: foods, t["col_tip"]: tips})
    st.markdown(f"### {t['plan_h']}")
    st.table(df)
