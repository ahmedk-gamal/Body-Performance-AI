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
    .stButton>button { 
        width: 100%; 
        border-radius: 5px; 
        height: 3em; 
        background-color: #ff4b4b; 
        color: white; 
        font-weight: bold;
        font-size: 18px;
    }
    .metric-card { 
        background-color: white; 
        padding: 20px; 
        border-radius: 10px; 
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1); 
    }
    th { background-color: #ff4b4b !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# 2. تحميل الموديل والـ Scaler
@st.cache_resource
def load_assets():
    # تأكد أن هذه الملفات موجودة في نفس مجلد الكود
    model = load_model('body_performance_model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"خطأ في تحميل ملفات الموديل: {e}")
    st.stop()

# 3. Sidebar - معلومات المشروع
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
st.markdown("قم بإدخال بياناتك الحيوية والرياضية للحصول على تصنيف دقيق لمستواك وبرنامجك المخصص.")

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

# 5. معالجة البيانات بعد الضغط على الزرار
if submit:
    # حساب القيم الإضافية (Feature Engineering)
    gender_val = 0 if gender == "ذكر" else 1
    bmi = weight / ((height / 100) ** 2)
    pulse_pressure = sys - dia
    
    # تجهيز البيانات للتوقع (تأكد من نفس ترتيب أعمدة التدريب)
    features = np.array([[age, gender_val, height, weight, fat, sys, dia, grip, sit_ups, bend, jump, bmi, pulse_pressure]])
    scaled_features = scaler.transform(features)
    
    # التوقع من الموديل
    prediction = model.predict(scaled_features)
    class_idx = np.argmax(prediction)
    
    classes = ['A (ممتاز)', 'B (جيد جداً)', 'C (جيد)', 'D (ضعيف)']
    result = classes[class_idx]
    
    # عرض النتائج الأساسية
    st.markdown("---")
    st.header("📊 النتيجة النهائية")
    
    col_res1, col_res2, col_res3 = st.columns(3)
    with col_res1:
        st.metric(label="التصنيف البدني", value=result)
    with col_res2:
        st.metric(label="مؤشر كتلة الجسم (BMI)", value=f"{bmi:.2f}")
    with col_res3:
        st.metric(label="فرق الضغط (Pulse Pressure)", value=pulse_pressure)
        
    if class_idx == 0:
        st.balloons()
        st.success("أداء استثنائي! أنت في أعلى مستويات اللياقة البدنية.")
    elif class_idx == 3:
        st.warning("مستواك البدني يحتاج لاهتمام عاجل وتغيير في نمط الحياة.")

    # 6. الجدول الأسبوعي المتكامل (تمرين + تغذية)
    st.markdown("---")
    st.header("📋 خطتك الأسبوعية المتكاملة")
    st.info("الجدول التالي مقترح بناءً على حالتك البدنية الحالية لضمان أفضل تطور.")

    # تعريف محتوى الجداول بناءً على الفئة
    if class_idx == 0: # فئة A
        data = {
            "اليوم": ["السبت", "الأحد", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة"],
            "التمارين": ["دفع (صدر/أكتاف/تراي)", "سحب (ظهر/باي)", "أرجل كاملة", "راحة إيجابية", "HIIT مكثف", "فول بادي قوة", "راحة تامة"],
            "تغذية (فواكه وخضروات)": ["موز وتوت", "تفاح وبروكلي", "أفوكادو وبنجر", "كيوي وخيار", "برتقال وكرفس", "رمان وبطاطا", "أي فاكهة"],
            "نصيحة غذائية": ["كارب عالي قبل التمرين", "بروتين مكثف", "معادن وأملاح", "ترطيب عالي", "مضادات أكسدة", "وجبة استشفاء", "يوم مفتوح"]
        }
    elif class_idx == 1: # فئة B
        data = {
            "اليوم": ["السبت", "الأحد", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة"],
            "التمارين": ["صدر + تراي", "ظهر + باي", "راحة", "أرجل + سمانة", "أكتاف + بطن", "جري 40 دقيقة", "راحة"],
            "تغذية (فواكه وخضروات)": ["تفاح وبروكلي", "موز وسبانخ", "فراولة وخيار", "برتقال وجزر", "أناناس وجرجير", "بطيخ وفلفل", "كيوي ولوز"],
            "نصيحة غذائية": ["بروتين متوسط", "ألياف عالية", "شرب مياه", "كارب معقد", "فيتامينات", "وجبة خفيفة", "راحة"]
        }
    elif class_idx == 2: # فئة C
        data = {
            "اليوم": ["السبت", "الأحد", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة"],
            "التمارين": ["مشي سريع + ضغط", "راحة", "تمارين وزن جسم", "مشي سريع", "راحة", "تمارين شاملة", "نشاط حر"],
            "تغذية (فواكه وخضروات)": ["كمثرى وفاصوليا", "تفاح وخيار", "موز وبروكلي", "جوافة وجزر", "توت وسلطة", "برتقال وطماطم", "بطيخ"],
            "نصيحة غذائية": ["منع السكريات", "مياه بانتظام", "بروتين أساسي", "خضار طازج", "تقليل ملح", "ورقيات خضراء", "توازن"]
        }
    else: # فئة D
        data = {
            "اليوم": ["السبت", "الأحد", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة"],
            "التمارين": ["مشي خفيف (20د)", "إطالة (15د)", "راحة", "مشي خفيف", "تقوية بسيطة", "مشي خفيف", "راحة"],
            "تغذية (فواكه وخضروات)": ["تفاح وخيار", "برتقال وخس", "جوافة وجزر", "موز وفلفل", "توت وجرجير", "كمثرى وفاصوليا", "سوتيه"],
            "نصيحة غذائية": ["منع مياه غازية", "ألياف عالية", "ترطيب مستمر", "فواكه قليلة سكر", "وجبات صغيرة", "شوربة خضار", "راحة"]
        }

    # عرض الجدول
    df = pd.DataFrame(data)
    st.table(df)

    st.warning("⚠️ **تنبيه:** هذا البرنامج استرشادي، يرجى استشارة مختص قبل البدء في تمارين عالية الشدة إذا كنت تعاني من إصابات.")

# 7. تذييل الصفحة
st.markdown("---")
st.caption("تم تطوير هذا النظام باستخدام تقنيات الذكاء الاصطناعي لتحسين جودة الحياة البدنية.")
