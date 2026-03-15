import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# 1. إعدادات الصفحة
st.set_page_config(page_title="Body Performance AI", page_icon="🏋️‍♂️", layout="centered")

# 2. تحميل الموديل والـ Scaler
@st.cache_resource
def load_assets():
    # تأكد أن هذه الملفات موجودة في نفس الـ Repository على GitHub
    model = load_model('body_performance_model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model/scaler: {e}")

# 3. دالة النصائح الذكية
def get_health_advice(category, bmi, fat_pct):
    st.markdown("---")
    st.subheader("📋 تقرير النصائح المخصصة (Health Advisory)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**تصنيف الأداء:** {category}")
        if category == 'A':
            st.success("أداء ممتاز! استمر في تمارين القوة والتحمل المتقدمة.")
        elif category == 'B':
            st.success("مستوى جيد جداً. ركز على تحسين مرونة الجسم.")
        elif category == 'C':
            st.warning("مستوى متوسط. تحتاج لزيادة عدد ساعات التمرين الأسبوعية.")
        else:
            st.error("مستوى ضعيف. ابدأ ببرنامج رياضي تدريجي تحت إشراف مختص.")

    with col2:
        st.info(f"**مؤشر كتلة الجسم (BMI):** {bmi:.2f}")
        if bmi < 18.5:
            st.warning("نحافة: يفضل زيادة السعرات الحرارية الصحية.")
        elif 18.5 <= bmi < 25:
            st.success("وزن مثالي: حافظ على نمط حياتك الحالي.")
        else:
            st.error("زيادة وزن: ركز على تمارين الكارديو وتقليل السكريات.")

# 4. واجهة المستخدم (UI)
st.title("🏋️‍♂️ نظام تقييم الأداء البدني الذكي")
st.write("أدخل بياناتك بالأسفل للحصول على تحليل دقيق لمستواك الرياضي نصائح لتحسينه.")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("العمر (Age)", min_value=1, max_value=100, value=25)
        gender = st.selectbox("النوع (Gender)", options=["ذكر (Male)", "أنثى (Female)"])
        height = st.number_input("الطول بالسنتيمتر (Height cm)", value=175.0)
        weight = st.number_input("الوزن بالكيلوجرام (Weight kg)", value=75.0)
        fat_pct = st.number_input("نسبة الدهون (Body Fat %)", value=20.0)
        diastolic = st.number_input("الضغط الانبساطي (Diastolic BP)", value=80.0)
        
    with col2:
        systolic = st.number_input("الضغط الانقباضي (Systolic BP)", value=120.0)
        grip = st.number_input("قوة القبضة (Grip Force)", value=40.0)
        sit_bend = st.number_input("اختبار الجلوس والتحريف (Sit & Bend)", value=15.0)
        sit_ups = st.number_input("تمارين البطن (Sit-ups count)", value=35)
        broad_jump = st.number_input("القفز الطويل (Broad Jump cm)", value=200.0)

    submit = st.form_submit_button("تحليل الأداء البدني 🚀")

# 5. معالجة البيانات والتوقع
if submit:
    # تحويل النوع لرقم (0 للأنثى، 1 للذكر)
    gender_val = 1 if "ذكر" in gender else 0
    
    # Feature Engineering (نفس اللي عملناه في التدريب)
    bmi = weight / ((height / 100) ** 2)
    pulse_pressure = systolic - diastolic
    
    # تجهيز الداتا للـ Scaler
    input_data = [age, gender_val, height, weight, fat_pct, diastolic, systolic, grip, sit_bend, sit_ups, broad_jump, bmi, pulse_pressure]
    
    # تحويل لـ DataFrame بنفس أسماء الأعمدة الأصلية
    columns = scaler.feature_names_in_
    input_df = pd.DataFrame([input_data], columns=columns)
    
    # الـ Scaling
    input_scaled = scaler.transform(input_df)
    
    # التوقع
    prediction_prob = model.predict(input_scaled)
    class_idx = np.argmax(prediction_prob)
    categories = ['A', 'B', 'C', 'D']
    result_class = categories[class_idx]
    
    # عرض النتيجة والنصيحة
    st.balloons()
    get_health_advice(result_class, bmi, fat_pct)
