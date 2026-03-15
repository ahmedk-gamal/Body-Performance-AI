import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# 1. إعدادات الصفحة الاحترافية
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
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    .metric-card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# 2. تحميل الموديل والـ Scaler
@st.cache_resource
def load_assets():
    model = load_model('body_performance_model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# 3. Sidebar - معلومات المشروع (الـ Cards اللي طلبتها)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3843/3843205.png", width=100)
    st.title("عن المشروع")
    
    with st.container():
        st.markdown("### 🤖 النموذج المستخدم")
        st.info("""
        **Multi-Layer Perceptron (MLP)**
        شبكة عصبية عميقة مكونة من 3 طبقات خفية مع 512-256-128 نيوترون.
        """)
        
    with st.container():
        st.markdown("### 📊 نبذة عن الموديل")
        st.write("""
        تم تدريب الموديل على أكثر من 13 ألف سجل حيوى. يستخدم الـ **Feature Engineering** لحساب الـ BMI و Pulse Pressure لزيادة دقة التصنيف.
        """)
    
    st.markdown("---")
    st.markdown("### 👨‍💻 المطور")
    st.success("**Ahmed Khaled Gamal**")

# 4. الواجهة الرئيسية
st.title("🧬 نظام تحليل الأداء البدني الذكي")
st.markdown("قم بإدخال بياناتك الحيوية والرياضية للحصول على تصنيف دقيق لمستواك.")

# تقسيم المدخلات في فورم منظم
with st.form("main_form"):
    t1, t2 = st.columns(2)
    
    with t1:
        st.subheader("📋 البيانات الأساسية")
        age = st.number_input("العمر", 10, 90, 25)
        gender = st.selectbox("النوع", ["ذكر", "أنثى"])
        height = st.number_input("الطول (سم)", 120.0, 220.0, 175.0)
        weight = st.number_input("الوزن (كجم)", 30.0, 200.0, 75.0)
        fat = st.number_input("نسبة دهون الجسم (%)", 5.0, 50.0, 20.0)

    with t2:
        st.subheader("💪 الاختبارات الرياضية")
        sys = st.number_input("الضغط الانقباضي", 80, 200, 120)
        dia = st.number_input("الضغط الانبساطي", 50, 120, 80)
        grip = st.number_input("قوة القبضة", 10.0, 100.0, 45.0)
        sit_ups = st.number_input("عدد تمارين البطن", 0
