
# 🏋️‍♂️ Body Performance AI: Deep Learning Classification & Advisory System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.17-orange?style=for-the-badge&logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Maintained%3F-Yes-green?style=for-the-badge" alt="Maintained">
</p>

---

## 🌟 نظرة عامة (Overview)
هذا المشروع ليس مجرد مصنف بيانات، بل هو **منصة ذكاء اصطناعي تفاعلية** تهدف إلى تقييم اللياقة البدنية للبشر. يقوم النظام بتحليل 11 متغيراً حيوياً ورياضياً، ثم يقوم بتصنيف الشخص إلى 4 فئات أداء (A, B, C, D) مع تقديم **خريطة طريق صحية** مخصصة لكل مستخدم.

---

## 🏗️ الهندسة التقنية (Technical Architecture)

### 🧠 الـ Model (المخ الذكي)
تم بناء شبكة عصبية عميقة من نوع **Multi-Layer Perceptron (MLP)** تتميز بـ:
* **Input Layer:** تستقبل 13 مدخلاً (بعد إضافة الميزات المبتكرة).
* **Hidden Layers:** ثلاث طبقات كثيفة (512 -> 256 -> 128) لضمان فهم العلاقات المعقدة بين البيانات.
* **Regularization:** استخدام **Dropout (0.5)** و **Early Stopping** لمنع الـ Overfitting وضمان دقة النتائج على البيانات الجديدة.
* **Optimizer:** استخدام `Adam` مع `ReduceLROnPlateau` لتسريع التعلم والوصول لأقل نسبة خطأ.



### 🧪 هندسة البيانات (Feature Engineering)
لرفع دقة الموديل من 68% إلى **75.2%**، قمنا بابتكار ميزات جديدة:
1.  **BMI (Body Mass Index):** لربط الوزن بالطول بشكل منطقي.
2.  **Pulse Pressure:** الفرق بين الضغط الانقباضي والانبساطي كمؤشر لقوة القلب البدنية.

---

## 💻 واجهة المستخدم (The Web App)
تم تطوير واجهة تفاعلية باستخدام **Streamlit** تسمح للمستخدم بـ:
* إدخال بياناته الحيوية (العمر، الوزن، نسبة الدهون، إلخ).
* الحصول على تصنيف فوري لمستواه البدني.
* قراءة تقرير نصائح "ديناميكي" يتغير بناءً على الـ BMI ونسبة الدهون.

---

## 🚀 كيفية التشغيل (Quick Start)

1. **تحميل المشروع:**
   ```bash
   git clone [https://github.com/ahmedk-gamal/body-performance-ai.git](https://github.com/ahmedk-gamal/body-performance-ai.git)
   cd body-performance-ai


   https://www.linkedin.com/in/k-ahmed-auc/
