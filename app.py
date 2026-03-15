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
    /* ألوان Apple Health */
    :root {
        --apple-red: #ff2d55;
        --apple-bg: #f2f2f7;
    }
    
    .main { background-color: var(--apple-bg); }
    
    /* تصميم الأزرار */
    .stButton>button { 
        width: 100%; border-radius: 12px; height: 3.5em; 
        background-color: #ff2d55; color: white; font-weight: 600; 
        font-size: 18px; border: none; transition: 0.3s;
        box-shadow: 0 4px 12px rgba(255, 45, 85, 0.2);
    }
    .stButton>button:hover {
        background-color: #e6264d;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 45, 85, 0.3);
    }

    /* تحسين شكل الجداول والكروت */
    div[data-testid="stMetricValue"] { color: #ff2d55; font-weight: 700; }
    .stTable { border-radius: 15px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    th { background-color: #ff2d55 !important; color: white !important; }
    
    /* إخفاء القوائم غير الضرورية لشكل أنظف */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 2. تحميل انيميشن القلب (Apple Health Style)
def load_lottieurl(url):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except: return None

# رابط قلب نابض أنيق
lottie_heart = load_lottieurl("https://lottie.host/82334807-6c2e-4054-8e36-7c0a6b72a6b2/v8Y8yO5x20.json")

# 3. نظام الترجمة والنبذة
with st.sidebar:
    if lottie_heart:
        st_lottie(lottie_heart, height=150, key="apple_heart")
    else:
        st.title("❤️ Health")
    
    lang = st.radio("🌐 Select Language", ("English", "العربية"))
    st.markdown("---")

texts = {
    "English": {
        "title": "🧬 Physical Health Analysis",
        "sub": "AI-powered performance tracking inspired by Apple Health.",
        "about_h": "🤖 Technology",
        "model_desc": "**MLP Neural Network**\nAdvanced predictive model for health classification.",
        "model_stats": "**Accuracy:** 82.19%\n**Dataset:** 13K+ records",
        "dev": "👨‍💻 Developer",
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
        "model_desc": "**الشبكة العصبية MLP**\nنموذج متطور للتنبؤ وتصنيف الحالة البدنية.",
        "model_stats": "**دقة النموذج:** 82.19%\n**البيانات:** 13 ألف+ سجل",
        "dev": "👨‍💻 المطور",
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

with st.sidebar:
    st.subheader(t["about_h"])
    st.info(t["model_desc"])
    st.caption(t["model_stats"])
    st.markdown("---")
    st.subheader(t["dev"])
    st.success(f"**Ahmed Khaled Gamal**")

# 4. تحميل الأصول التقنية
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

# 5. التصميم الرئيسي
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

# 6. المعالجة وعرض النتائج
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

    # جداول الخطة (كما في النسخ السابقة مع تحسين المحتوى)
    if lang == "English":
        workouts = {0:["Push (Chest/Shoulders)","Pull (Back/Biceps)","Legs (Quads/Glutes)","Active Recovery","HIIT Session","Full Body Power","Full Rest"],
                    1:["Chest/Triceps","Back/Biceps","Full Rest","Lower Body","Shoulders/Abs","Cardio (40 min)","Rest"],
                    2:["Power Walking","Rest","Bodyweight Skills","Power Walking","Rest","Full Body Circuit","Active Day"],
                    3:["Light Walk (20 min)","Static Stretching","Rest","Light Walk","Gentle Yoga","Light Walk","Rest"]}
        foods = ["Berries & Spinach","Apples & Broccoli","Avocado & Beets","Kiwi & Cucumber","Citrus & Carrots","Pomegranate","Seasonal Mix"]
        tips = ["High Carb (Pre)","Lean Protein","Hydration (3L)","Complex Carbs","Antioxidants","Muscle Repair","Relax"]
    else:
        workouts = {0:["دفع (صدر وأكتاف)","سحب (ظهر وباي)","أرجل (عضلات أمامية)","استشفاء نشط","تمرين HIIT","قوة شاملة","راحة تامة"],
                    1:["صدر وتراي","ظهر وباي","راحة","أرجل كاملة","أكتاف وبطن","كارديو (40 د)","راحة"],
                    2:["مشي سريع","راحة","تمارين وزن الجسم","مشي سريع","راحة","دائرة تدريبية","يوم نشط"],
                    3:["مشي خفيف (20 د)","إطالات ثابتة","راحة","مشي خفيف","يوجا بسيطة","مشي خفيف","راحة"]}
        foods = ["توت وسبانخ","تفاح وبروكلي","أفوكادو وبنجر","كيوي وخيار","حمضيات وجزر","رمان","فاكهة الموسم"]
        tips = ["كارب عالي","بروتين صافي","ترطيب (3 لتر)","كارب معقد","مضادات أكسدة","ترميم العضلات","استرخاء"]

    df = pd.DataFrame({t["col_day"]: t["days"], t["col_exe"]: workouts[idx], t["col_nut"]: foods, t["col_tip"]: tips})
    st.markdown(f"### {t['plan_h']}")
    st.table(df)
