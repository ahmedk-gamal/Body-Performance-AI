# 🏋️‍♂️ Body Performance AI: Deep Learning Classification & Advisory System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.17-orange?style=for-the-badge&logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Streamlit-Live-red?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Maintained%3F-Yes-green?style=for-the-badge" alt="Maintained">
</p>

---

## 🌟 نظرة عامة (Overview)
هذا المشروع هو منصة ذكاء اصطناعي تفاعلية لتقييم اللياقة البدنية، تم تطويره كجزء من دمج علوم البيانات في تحليل الأداء البشري. يقوم النظام بتحليل القياسات الحيوية والرياضية لتصنيف المستخدم إلى 4 فئات أداء (A, B, C, D) مع تقديم **توصيات صحية ورياضية مخصصة** بناءً على النتيجة ومؤشر كتلة الجسم (BMI).

---

## 🏗️ Technical Architecture

### 🧠 بنية النموذج (Model Architecture)
تم بناء شبكة عصبية عميقة من نوع **Multi-Layer Perceptron (MLP)** تتميز بـ:
* **Input Layer:** تستقبل 13 ميزة (بعد إضافة الميزات المبتكرة).
* **Hidden Layers:** ثلاث طبقات كثيفة (512 -> 256 -> 128) مع وظيفة تنشيط `ReLU`.
* **Regularization:** استخدام **Dropout (0.5)** و **Early Stopping** لضمان استقرار النموذج وتجنب الـ Overfitting.
* **Optimizer:** المحسن `Adam` مع تقنية `ReduceLROnPlateau` للوصول لأفضل دقة ممكنة.

![Model Architecture](https://www.researchgate.net/publication/344464132/figure/fig1/AS:942547190022144@1601729019681/A-simple-structure-of-a-Multi-layer-Perceptron-MLP-neural-network.png)

### 🧪 Feature Engineering
تم تحسين أداء النموذج من خلال ابتكار مؤشرات إضافية:
1.  **BMI (Body Mass Index):** لربط الوزن بالطول وتوفير سياق أعمق للموديل.
2.  **Pulse Pressure:** الفرق بين الضغط الانقباضي والانبساطي كمؤشر لقوة القلب البدنية.

---

## 💻 واجهة المستخدم (The Web App)
التطبيق متاح عبر الويب باستخدام **Streamlit**، حيث يتيح:
- إدخال البيانات الحيوية بسهولة.
- الحصول على تصنيف فوري للأداء البدني.
- عرض نصائح ديناميكية تعتمد على حالة المستخدم الصحية (نسبة الدهون والـ BMI).

---

## 🚀 كيفية التشغيل (Quick Start)

1. **تحميل المشروع:**
   ```bash
   git clone [https://github.com/ahmedk-gamal/body-performance-ai.git](https://github.com/ahmedk-gamal/body-performance-ai.git)
   cd body-performance-ai





تثبيت المكتبات:


pip install -r requirements.txt


تشغيل التطبيق:


streamlit run app.py





📊 البيانات (Dataset)
مجموعة بيانات ضخمة تحتوي على 13,393 سجل، تشمل اختبارات رياضية مثل (قوة القبضة، تمارين البطن، القفز الطويل) وقياسات حيوية (ضغط الدم، نسبة الدهون).

🔮 الرؤية المستقبلية (Future Roadmap)
[ ] إضافة دعم للغة العربية بالكامل.

[ ] ربط النظام بتقنيات رؤية الكمبيوتر (Computer Vision) لتصحيح وضعيات التمرين.

[ ] تطوير لوحة تحكم (Dashboard) لمتابعة التطور الزمني لأداء المستخدم.

👨‍💼 صاحب المشروع (Author)
Ahmed Khaled Gamal Hossam Eldien

⭐️ إذا أعجبك هذا العمل، لا تنسَ دعم المشروع بـ Star على GitHub!




