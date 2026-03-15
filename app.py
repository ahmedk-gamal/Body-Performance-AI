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
        sit_ups = st.number_input("عدد تمارين البطن", 0, 100, 35)
        jump = st.number_input("القفز الطويل (سم)", 50.0, 350.0, 210.0)
        bend = st.number_input("مرونة الظهر (Sit & Bend)", -20.0, 50.0, 15.0)

    submit = st.form_submit_button("تحليل البيانات الآن ✨")



if submit:
    # 1. تحويل النوع لقيمة رقمية (زي ما الموديل متدرب)
    gender_val = 0 if gender == "ذكر" else 1
    
    # 2. حساب الـ Feature Engineering (اللي أنت كاتبه في السايد بار)
    bmi = weight / ((height / 100) ** 2)
    pulse_pressure = sys - dia
    
    # 3. تجهيز البيانات في مصفوفة (تأكد من ترتيب الأعمدة كما في التدريب)
    # الترتيب ده افتراضي، رتبه حسب الـ Features اللي الموديل متعود عليها
    features = np.array([[age, gender_val, height, weight, fat, sys, dia, grip, sit_ups, bend, jump, bmi, pulse_pressure]])
    
    # 4. عمل Scaling للبيانات
    scaled_features = scaler.transform(features)
    
    # 5. التوقع (Prediction)
    prediction = model.predict(scaled_features)
    class_idx = np.argmax(prediction)
    
    # 6. تحويل النتيجة لكلام مفهوم (A, B, C, D)
    classes = ['A (ممتاز)', 'B (جيد جداً)', 'C (جيد)', 'D (ضعيف)']
    result = classes[class_idx]
    
    # 7. عرض النتيجة بشكل شيك
    st.markdown("---")
    st.header("📊 النتيجة النهائية")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="التصنيف البدني", value=result)
    with col2:
        st.metric(label="مؤشر كتلة الجسم (BMI)", value=f"{bmi:.2f}")
        
    if class_idx == 0:
        st.balloons()
        st.success("أداء استثنائي! استمر في المحافظة على مستواك.")
    elif class_idx == 3:
        st.warning("مستواك البدني محتاج تطوير، ابدأ ببرنامج رياضي تدريجي.")




# (بعد عرض النتيجة مباشرة)
    st.markdown("---")
    st.subheader("💡 نصائح الخبراء المخصصة لك")
    
    # مصفوفة النصائح بناءً على الفئة
    recommendations = {
        0: { # فئة A
            "title": "أنت في القمة! كيف تحافظ عليها؟",
            "tips": ["ركز على تمارين الـ Hypertrophy لزيادة الكتلة العضلية.", "جرب تمارين الـ HIIT لرفع كفاءة القلب.", "اهتم بفترات الاستشفاء (Recovery) والنوم العميق."],
            "article": "https://www.healthline.com/health/fitness-exercise/staying-fit",
            "color": "green"
        },
        1: { # فئة B
            "title": "أداء رائع.. خطوات بسيطة للوصول للمثالية",
            "tips": ["زود شدة التمارين (Intensity) بنسبة 10% أسبوعياً.", "ركز على مرونة الظهر وتمارين الإطالة.", "تأكد من تناول كمية كافية من البروتين (1.6 جم لكل كيلو)."],
            "article": "https://www.verywellfit.com/moving-from-intermediate-to-advanced-workout-1231225",
            "color": "blue"
        },
        2: { # فئة C
            "title": "بداية جيدة.. نحتاج لزيادة الالتزام",
            "tips": ["التزم بـ 3 أيام تدريب أسبوعياً كحد أدنى.", "حسن عاداتك الغذائية وقلل السكريات.", "ركز على الكارديو (المشي السريع أو الجري) لمدة 30 دقيقة."],
            "article": "https://www.nerdfitness.com/blog/how-to-build-your-own-workout-routine/",
            "color": "orange"
        },
        3: { # فئة D
            "title": "خطة الإنقاذ البدني.. ابدأ الآن!",
            "tips": ["ابدأ بتمارين وزن الجسم (ضغط، سكوات) في المنزل.", "المشي لمدة 20 دقيقة يومياً سيحدث فرقاً شاسعاً.", "استشر طبيب تغذية لتقليل نسبة الدهون بأمان."],
            "article": "https://www.healthline.com/health/fitness-exercise/how-to-start-working-out",
            "color": "red"
        }
    }

    user_rec = recommendations[class_idx]
    
    # عرض النصائح في Container ملون
    with st.container():
        st.markdown(f"#### {user_rec['title']}")
        for tip in user_rec['tips']:
            st.write(f"- {tip}")
        
        st.markdown(f"[🔗 اقرأ مقالاً تفصيلياً لنظامك المثالي من هنا]({user_rec['article']})")
