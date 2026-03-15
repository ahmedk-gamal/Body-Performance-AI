import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# 1. إعدادات الصفحة
st.set_page_config(
    page_title="Body Performance AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تحسين المظهر باستخدام CSS
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { 
        width: 100%; border-radius: 5px; height: 3em; 
        background-color: #ff4b4b; color: white; font-weight: bold; font-size: 18px;
    }
    th { background-color: #ff4b4b !important; color: white !important; text-align: center !important; }
    td { text-align: center !important; }
    </style>
    """, unsafe_allow_html=True)

# 2. نظام الترجمة (Translation System)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3843/3843205.png", width=100)
    lang = st.radio("🌐 Language / اللغة", ("English", "العربية"))
    st.markdown("---")

# قاموس النصوص
texts = {
    "English": {
        "title": "🧬 AI Body Performance System",
        "sub": "Enter your data for a personalized fitness class and plan.",
        "basic_h": "📋 Basic Information",
        "age": "Age", "gender": "Gender", "male": "Male", "female": "Female",
        "height": "Height (cm)", "weight": "Weight (kg)", "fat": "Body Fat %",
        "test_h": "💪 Fitness Tests",
        "sys": "Systolic BP", "dia": "Diastolic BP", "grip": "Grip Force",
        "situps": "Sit-ups", "jump": "Broad Jump (cm)", "bend": "Sit & Bend",
        "btn": "Analyze Data Now ✨",
        "res_h": "📊 Final Result",
        "class": "Classification", "bmi": "BMI Index", "pp": "Pulse Pressure",
        "plan_h": "🗓️ Weekly Integrated Plan",
        "col_day": "Day", "col_exe": "Exercises", "col_nut": "Fruits/Veggies", "col_tip": "Nutrition Tip",
        "days": ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"],
        "classes": ['A (Excellent)', 'B (Good)', 'C (Fair)', 'D (Poor)'],
        "footer": "The Greatest Wealth is Health"
    },
    "العربية": {
        "title": "🧬 نظام تحليل الأداء البدني الذكي",
        "sub": "أدخل بياناتك للحصول على تصنيف بدني وخطة تدريب مخصصة.",
        "basic_h": "📋 البيانات الأساسية",
        "age": "العمر", "gender": "النوع", "male": "ذكر", "female": "أنثى",
        "height": "الطول (سم)", "weight": "الوزن (كجم)", "fat": "نسبة الدهون %",
        "test_h": "💪 الاختبارات الرياضية",
        "sys": "الضغط الانقباضي", "dia": "الضغط الانبساطي", "grip": "قوة القبضة",
        "situps": "تمارين البطن", "jump": "القفز الطويل (سم)", "bend": "مرونة الظهر",
        "btn": "تحليل البيانات الآن ✨",
        "res_h": "📊 النتيجة النهائية",
        "class": "التصنيف البدني", "bmi": "مؤشر كتلة الجسم", "pp": "فرق الضغط",
        "plan_h": "🗓️ خطتك الأسبوعية المتكاملة",
        "col_day": "اليوم", "col_exe": "التمارين", "col_nut": "فواكه وخضروات", "col_tip": "نصيحة التغذية",
        "days": ["السبت", "الأحد", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة"],
        "classes": ['A (ممتاز)', 'B (جيد جداً)', 'C (جيد)', 'D (ضعيف)'],
        "footer": "الصحة هي الثروة الحقيقية"
    }
}
t = texts[lang]

# 3. تحميل الموديل والـ Scaler
@st.cache_resource
def load_assets():
    model = load_model('body_performance_model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 4. الواجهة الرئيسية
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

# 5. النتائج والجدول
if submit:
    gender_val = 0 if gender == t["male"] else 1
    bmi = weight / ((height / 100) ** 2)
    pulse_pressure = sys - dia
    features = np.array([[age, gender_val, height, weight, fat, sys, dia, grip, sit_ups, bend, jump, bmi, pulse_pressure]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    class_idx = np.argmax(prediction)
    
    st.markdown("---")
    st.header(t["res_h"])
    c_res = st.columns(3)
    c_res[0].metric(t["class"], t["classes"][class_idx])
    c_res[1].metric(t["bmi"], f"{bmi:.2f}")
    c_res[2].metric(t["pp"], pulse_pressure)

    # بناء بيانات الجدول بناءً على الفئة واللغة
    if lang == "English":
        plans = {
            0: ["Push Day", "Pull Day", "Legs", "Active Rest", "HIIT", "Full Body", "Rest"],
            1: ["Chest/Tri", "Back/Bi", "Rest", "Legs", "Shoulders", "Cardio", "Rest"],
            2: ["Fast Walk", "Rest", "Bodyweight", "Fast Walk", "Rest", "Full Body", "Active"],
            3: ["Light Walk", "Stretch", "Rest", "Light Walk", "Easy Cardio", "Light Walk", "Rest"]
        }
        nut = ["Banana/Berries", "Apple/Broccoli", "Avocado", "Kiwi", "Orange", "Pomegranate", "Seasonal"]
        tips = ["High Carb", "High Protein", "Hydrate", "Vitamins", "Antioxidants", "Recovery", "Cheat Day"]
    else:
        plans = {
            0: ["دفع (صدر/أكتاف)", "سحب (ظهر/باي)", "أرجل", "راحة إيجابية", "HIIT", "فول بادي", "راحة"],
            1: ["صدر + تراي", "ظهر + باي", "راحة", "أرجل", "أكتاف", "جري 40د", "راحة"],
            2: ["مشي سريع", "راحة", "وزن جسم", "مشي سريع", "راحة", "تمارين شاملة", "نشاط حر"],
            3: ["مشي خفيف", "إطالة", "راحة", "مشي خفيف", "تقوية بسيطة", "مشي خفيف", "راحة"]
        }
        nut = ["موز وتوت", "تفاح وبروكلي", "أفوكادو", "كيوي", "برتقال", "رمان", "فاكهة موسمية"]
        tips = ["كارب عالي", "بروتين مكثف", "ترطيب", "فيتامينات", "مضادات أكسدة", "استشفاء", "يوم مفتوح"]

    df_data = {
        t["col_day"]: t["days"],
        t["col_exe"]: plans[class_idx],
        t["col_nut"]: nut,
        t["col_tip"]: tips
    }
    st.markdown(f"### {t['plan_h']}")
    st.table(pd.DataFrame(df_data))

st.markdown("---")
st.caption(f"© 2026 Ahmed Khaled Gamal - {t['footer']}")
