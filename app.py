import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from streamlit_lottie import st_lottie # مكتبة الانيميشن الجديدة
import requests # عشان نجيب الانيميشن من النت

# 1. إعدادات الصفحة الاحترافية
st.set_page_config(
    page_title="Body Performance AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 تحسين الـ CSS ليدعم الـ Dark & Light Mode وألوان متناسقة
st.markdown("""
    <style>
    /* تغيير لون الزرار وتأثيره عند الوقوف عليه */
    .stButton>button { 
        width: 100%; border-radius: 8px; height: 3.5em; 
        background-color: #ff4b4b; color: white; font-weight: bold; 
        font-size: 18px; border: none; transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
    }
    
    /* تنسيق الجداول لتكون احترافية */
    th { text-align: center !important; }
    td { text-align: center !important; }
    
    /* تنسيق الـ Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. دالة لتحميل انيميشن Lottie (القلب النابض)
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# رابط انيميشن قلب بينبض (ممكن تغيره برابط JSON تاني لو حبيت)
lottie_heart = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_9czw5u1f.json")

# 3. نظام الترجمة والنبذة التقنية (كما في النسخة السابقة)
with st.sidebar:
    # عرض القلب النابض بدل الصورة الثابتة
    if lottie_heart:
        st_lottie(lottie_heart, height=120, key="heart")
    else:
        st.write("❤️") # fallback في حالة فشل التحميل

    lang = st.radio("🌐 Language / اللغة", ("English", "العربية"))
    st.markdown("---")

texts = {
    "English": {
        "title": "🧬 AI Body Performance System",
        "sub": "Predict your fitness class and get a custom plan.",
        "about_h": "🤖 About the Model",
        "model_desc": "**Multi-Layer Perceptron (MLP)**\nA deep neural network with 3 hidden layers (512-256-128 neurons).",
        "model_stats": "**Model Accuracy:** 82.19%\n**Dataset:** 13,000+ biological records.",
        "dev": "👨‍💻 Developer",
        "basic_h": "📋 Vital Data",
        "age": "Age", "gender": "Gender", "male": "Male", "female": "Female",
        "height": "Height (cm)", "weight": "Weight (kg)", "fat": "Body Fat %",
        "test_h": "💪 Performance Tests",
        "sys": "Systolic", "dia": "Diastolic", "grip": "Grip Force",
        "situps": "Sit-ups", "jump": "Jump (cm)", "bend": "Flexibility",
        "btn": "Analyze Now ✨",
        "res_h": "📊 Results",
        "class": "Classification", "bmi": "BMI Index", "pp": "Pulse Pressure",
        "plan_h": "🗓️ Your Weekly Plan",
        "col_day": "Day", "col_exe": "Workout", "col_nut": "Veggies/Fruit", "col_tip": "Tip",
        "days": ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"],
        "classes": ['A (Excellent)', 'B (Good)', 'C (Fair)', 'D (Poor)']
    },
    "العربية": {
        "title": "🧬 نظام تحليل الأداء البدني الذكي",
        "sub": "توقع فئتك البدنية واحصل على خطتك التدريبية المخصصة.",
        "about_h": "🤖 عن النموذج المستخدم",
        "model_desc": "**Perceptron متعدد الطبقات (MLP)**\nشبكة عصبية عميقة مكونة من 3 طبقات خفية (512-256-128 نيوترون).",
        "model_stats": "**دقة النموذج:** 82.19%\n**البيانات:** أكثر من 13 ألف سجل حيوي.",
        "dev": "👨‍💻 المطور",
        "basic_h": "📋 البيانات الحيوية",
        "age": "العمر", "gender": "النوع", "male": "ذكر", "female": "أنثى",
        "height": "الطول (سم)", "weight": "الوزن (كجم)", "fat": "نسبة الدهون %",
        "test_h": "💪 اختبارات الأداء",
        "sys": "الضغط الانقباضي", "dia": "الضغط الانبساطي", "grip": "قوة القبضة",
        "situps": "تمارين البطن", "jump": "القفز (سم)", "bend": "المرونة",
        "btn": "حلل الآن ✨",
        "res_h": "📊 النتائج",
        "class": "التصنيف البدني", "bmi": "مؤشر الكتلة", "pp": "فرق الضغط",
        "plan_h": "🗓️ خطتك الأسبوعية المتكاملة",
        "col_day": "اليوم", "col_exe": "التمرين", "col_nut": "الخضار/الفاكهة", "col_tip": "نصيحة",
        "days": ["السبت", "الأحد", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة"],
        "classes": ['A (ممتاز)', 'B (جيد جداً)', 'C (جيد)', 'D (ضعيف)']
    }
}
t = texts[lang]

# عرض النبذة التقنية والمطور في السايد بار بناءً على اللغة
with st.sidebar:
    st.subheader(t["about_h"])
    st.info(t["model_desc"])
    st.write(t["model_stats"])
    st.markdown("---")
    st.subheader(t["dev"])
    st.success("**Ahmed Khaled Gamal**")

# 4. تحميل الموديل والـ Scaler (مع معالجة الأخطاء)
@st.cache_resource
def load_assets():
    try:
        model = load_model('body_performance_model.keras')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_assets()

if model is None:
    st.error("Error: Could not load model or scaler files. Ensure 'body_performance_model.keras' and 'scaler.pkl' are in the same directory.")
    st.stop()

# 5. الواجهة الرئيسية والمدخلات
st.title(t["title"])
st.markdown(t["sub"])

with st.form("main_form"):
    t1, t2 = st.columns(2)
    with t1:
        st.subheader(t["basic_h"])
        age = st.number_input(t["age"], 10, 90, 25)
        gender = st.selectbox(t["gender"], [t["male"], t["female"]])
        height = st.number_input(t["height"], 120.0, 220.0, 175.0)
        weight = st.number_input(t["weight"], 30.0, 200.0, 75.0)
        fat = st.number_input(t["fat"], 5.0, 50.0, 20.0)
    with t2:
        st.subheader(t["test_h"])
        sys = st.number_input(t["sys"], 80, 200, 120)
        dia = st.number_input(t["dia"], 50, 120, 80)
        grip = st.number_input(t["grip"], 10.0, 100.0, 45.0)
        sit_ups = st.number_input(t["situps"], 0, 100, 35)
        jump = st.number_input(t["jump"], 50.0, 350.0, 210.0)
        bend = st.number_input(t["bend"], -20.0, 50.0, 15.0)
    submit = st.form_submit_button(t["btn"])

# 6. المعالجة وعرض النتائج
if submit:
    # Feature Engineering
    gender_val = 0 if gender == t["male"] else 1
    bmi = weight / ((height / 100) ** 2)
    pulse_pressure = sys - dia
    
    # تحضير البيانات للتوقع
    features = np.array([[age, gender_val, height, weight, fat, sys, dia, grip, sit_ups, bend, jump, bmi, pulse_pressure]])
    scaled_features = scaler.transform(features)
    
    # التوقع
    prediction = model.predict(scaled_features)
    class_idx = np.argmax(prediction)
    
    st.markdown("---")
    st.header(t["res_h"])
    c_res = st.columns(3)
    c_res[0].metric(t["class"], t["classes"][class_idx])
    c_res[1].metric(t["bmi"], f"{bmi:.2f}")
    c_res[2].metric(t["pp"], pulse_pressure)

    # بيانات الجدول بناءً على الفئة واللغة
    if lang == "English":
        plans = {0:["Push Day","Pull Day","Legs","Active Rest","HIIT","Full Body","Rest"],
                 1:["Chest/Tri","Back/Bi","Rest","Legs","Shoulders","Cardio","Rest"],
                 2:["Fast Walk","Rest","Bodyweight","Fast Walk","Rest","Full Body","Active"],
                 3:["Light Walk","Stretch","Rest","Light Walk","Easy Cardio","Light Walk","Rest"]}
        nut = ["Banana/Berries","Apple/Broccoli","Avocado","Kiwi","Orange","Pomegranate","Seasonal"]
        tips = ["High Carb","High Protein","Hydrate","Vitamins","Antioxidants","Recovery","Cheat Day"]
    else:
        plans = {0:["دفع (صدر/أكتاف)","سحب (ظهر/باي)","أرجل","راحة إيجابية","HIIT","فول بادي","راحة"],
                 1:["صدر + تراي","ظهر + باي","راحة","أرجل","أكتاف","جري 40د","راحة"],
                 2:["مشي سريع","راحة","وزن جسم","مشي سريع","راحة","تمارين شاملة","نشاط حر"],
                 3:["مشي خفيف","إطالة","راحة","مشي خفيف","تقوية بسيطة","مشي خفيف","راحة"]}
        nut = ["موز وتوت","تفاح وبروكلي","أفوكادو","كيوي","برتقال","رمان","فاكهة موسمية"]
        tips = ["كارب عالي","بروتين مكثف","ترطيب","فيتامينات","مضادات أكسدة","استشفاء","يوم مفتوح"]

    df_data = {t["col_day"]: t["days"], t["col_exe"]: plans[class_idx], t["col_nut"]: nut, t["col_tip"]: tips}
    st.markdown(f"### {t['plan_h']}")
    st.table(pd.DataFrame(df_data))
