import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from streamlit_lottie import st_lottie
import requests
import io
from datetime import datetime

# ==========================================
# 1. إعدادات الصفحة
# ==========================================
st.set_page_config(
    page_title="Health AI | Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. التنسيق الجمالي
# ==========================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    :root {
        --apple-red: #ff2d55;
        --soft-blue: #f0f7ff;
        --deep-blue: #1a237e;
    }
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }
    .main-header-container {
        display: flex; flex-direction: column;
        align-items: center; justify-content: center; margin-top: -30px;
    }
    .main-title-text {
        color: var(--deep-blue); font-size: 42px; font-weight: 800;
        text-align: center; margin-top: -10px; margin-bottom: 20px;
    }
    .sidebar-header {
        display: flex; flex-direction: column;
        align-items: center; justify-content: center; margin-bottom: 20px;
    }
    .logo-text {
        color: var(--apple-red); font-size: 28px; font-weight: 800;
        letter-spacing: 2px; text-align: center;
    }
    .stButton>button {
        width: 100%; border-radius: 12px; height: 3.8em;
        background-color: var(--apple-red); color: white; font-weight: 700;
        border: none; transition: all 0.3s ease; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: #e6264d; transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .about-box {
        background-color: var(--soft-blue); padding: 20px; border-radius: 15px;
        border-left: 6px solid #2196F3; font-size: 14px; color: var(--deep-blue);
        margin-bottom: 25px; line-height: 1.6;
    }
    .social-link {
        display: flex; align-items: center; gap: 12px;
        text-decoration: none; font-weight: 700; margin-bottom: 15px; transition: 0.3s;
    }
    .linkedin { color: #0077b5 !important; }
    .github   { color: #333 !important; }
    th { background-color: var(--apple-red) !important; color: white !important; text-align: center !important; }
    [data-testid="stMetricValue"] { color: var(--apple-red); font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. دالة PDF الكاملة بـ reportlab
# ==========================================
def create_pdf(summary, plan_df=None, fitness_class_idx=0):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer,
        Table, TableStyle, HRFlowable
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.graphics.shapes import Drawing, Rect, String, Circle, Wedge, Line
    from reportlab.graphics import renderPDF
    from reportlab.graphics.charts.piecharts import Pie
    import math
    import qrcode
    from PIL import Image as PILImage
    from reportlab.platypus import Image as RLImage

    # ===== الألوان =====
    RED   = colors.HexColor("#ff2d55")
    BLUE  = colors.HexColor("#1a237e")
    LBLUE = colors.HexColor("#f0f7ff")
    GRAY  = colors.HexColor("#f5f5f5")
    WHITE = colors.white
    DARK  = colors.HexColor("#222222")
    GREEN = colors.HexColor("#4caf50")
    ORANGE= colors.HexColor("#ff9800")
    PURPLE= colors.HexColor("#9c27b0")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=1.8*cm, leftMargin=1.8*cm,
        topMargin=1.8*cm,   bottomMargin=1.8*cm
    )

    styles  = getSampleStyleSheet()
    W = 17.4 * cm   # عرض المحتوى

    # ===== Styles =====
    title_style = ParagraphStyle(
        "MyTitle", fontSize=20, textColor=WHITE,
        alignment=TA_CENTER, fontName="Helvetica-Bold", leading=24
    )
    sub_style = ParagraphStyle(
        "MySub", fontSize=9, textColor=colors.HexColor("#cccccc"),
        alignment=TA_CENTER, fontName="Helvetica", spaceAfter=0
    )
    section_style = ParagraphStyle(
        "MySection", fontSize=13, textColor=BLUE,
        spaceBefore=12, spaceAfter=6, fontName="Helvetica-Bold"
    )
    footer_style = ParagraphStyle(
        "MyFooter", fontSize=8, textColor=colors.gray,
        alignment=TA_CENTER, fontName="Helvetica"
    )

    story = []

    # ==========================================
    # HEADER BANNER
    # ==========================================
    now_str = datetime.now().strftime("%Y-%m-%d  %H:%M")

    header_content = [
        [Paragraph("HEALTH AI  |  OFFICIAL HEALTH REPORT", title_style)],
        [Paragraph(f"Generated: {now_str}  |  Developer: Ahmed Khaled Gamal", sub_style)],
    ]
    header_table = Table(header_content, colWidths=[W])
    header_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), RED),
        ("TOPPADDING",    (0,0), (-1, 0), 14),
        ("BOTTOMPADDING", (0,0), (-1, 0), 4),
        ("TOPPADDING",    (0,1), (-1,-1), 2),
        ("BOTTOMPADDING", (0,1), (-1,-1), 12),
        ("LEFTPADDING",   (0,0), (-1,-1), 16),
        ("RIGHTPADDING",  (0,0), (-1,-1), 16),
        ("ROUNDEDCORNERS",(0,0), (-1,-1), [8,8,8,8]),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.4*cm))

    # ==========================================
    # SECTION 1: HEALTH METRICS + GAUGE BMI
    # ==========================================
    story.append(Paragraph("Health Metrics", section_style))
    story.append(HRFlowable(width=W, thickness=1.5, color=RED, spaceAfter=6))

    # --- Gauge Chart للـ BMI ---
    def make_bmi_gauge(bmi_val):
        d = Drawing(160, 110)

        cx, cy, r = 80, 55, 55
        # خلفية رمادية
        d.add(Wedge(cx, cy, r, 180, 360, fillColor=colors.HexColor("#eeeeee"), strokeColor=None))

        # Zones
        zones = [
            (180, 210, colors.HexColor("#2196f3")),   # Underweight
            (210, 252, colors.HexColor("#4caf50")),   # Normal
            (252, 288, colors.HexColor("#ff9800")),   # Overweight
            (288, 360, colors.HexColor("#f44336")),   # Obese
        ]
        for start, end, c in zones:
            d.add(Wedge(cx, cy, r, start, end, fillColor=c, strokeColor=WHITE, strokeWidth=1))

        # Inner white circle
        d.add(Circle(cx, cy, r*0.55, fillColor=WHITE, strokeColor=None))

        # Needle
        bmi_clamped = max(15, min(bmi_val, 40))
        angle_deg = 180 + ((bmi_clamped - 15) / 25) * 180
        angle_rad = math.radians(angle_deg)
        nx = cx + (r * 0.48) * math.cos(angle_rad)
        ny = cy + (r * 0.48) * math.sin(angle_rad)
        d.add(Line(cx, cy, nx, ny, strokeColor=DARK, strokeWidth=2.5))
        d.add(Circle(cx, cy, 5, fillColor=DARK, strokeColor=None))

        # BMI value text
        d.add(String(cx, cy - 20, f"{bmi_val:.1f}", fontSize=13,
                     fillColor=DARK, textAnchor='middle', fontName="Helvetica-Bold"))
        d.add(String(cx, cy - 32, "BMI", fontSize=9,
                     fillColor=colors.gray, textAnchor='middle', fontName="Helvetica"))

        # Labels
        labels = [("15", 180), ("Normal", 225), ("30", 270), ("40", 360)]
        for lbl, ang in [("15", 182), ("30", 270), ("40", 358)]:
            ar = math.radians(ang)
            lx = cx + (r + 8) * math.cos(ar)
            ly = cy + (r + 8) * math.sin(ar)
            d.add(String(lx, ly, lbl, fontSize=7, fillColor=colors.gray,
                         textAnchor='middle', fontName="Helvetica"))

        return d

    bmi_val = float(summary.get("BMI Score", 22))

    # BMI category label
    if bmi_val < 18.5:
        bmi_cat, bmi_color = "Underweight", colors.HexColor("#2196f3")
    elif bmi_val < 25:
        bmi_cat, bmi_color = "Normal", colors.HexColor("#4caf50")
    elif bmi_val < 30:
        bmi_cat, bmi_color = "Overweight", colors.HexColor("#ff9800")
    else:
        bmi_cat, bmi_color = "Obese", colors.HexColor("#f44336")

    bmi_cat_style = ParagraphStyle(
        "BmiCat", fontSize=10, textColor=bmi_color,
        alignment=TA_CENTER, fontName="Helvetica-Bold"
    )

    gauge_drawing = make_bmi_gauge(bmi_val)

    # --- Metrics Table ---
    metrics_data = [["Metric", "Value"]]
    for k, v in summary.items():
        metrics_data.append([str(k), str(v)])

    metrics_table = Table(metrics_data, colWidths=[6*cm, 5.5*cm])
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), RED),
        ("TEXTCOLOR",     (0,0), (-1,0), WHITE),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,0), 10),
        ("ALIGN",         (0,0), (-1,0), "CENTER"),
        ("TOPPADDING",    (0,0), (-1,0), 7),
        ("BOTTOMPADDING", (0,0), (-1,0), 7),
        ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",      (0,1), (-1,-1), 9),
        ("TEXTCOLOR",     (0,1), (0,-1), BLUE),
        ("TEXTCOLOR",     (1,1), (1,-1), RED),
        ("FONTNAME",      (0,1), (0,-1), "Helvetica-Bold"),
        ("ALIGN",         (0,0), (-1,-1), "LEFT"),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("TOPPADDING",    (0,1), (-1,-1), 6),
        ("BOTTOMPADDING", (0,1), (-1,-1), 6),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [LBLUE, WHITE]),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#e0e0e0")),
    ]))

    gauge_cell = Table(
        [[gauge_drawing], [Paragraph(bmi_cat, bmi_cat_style)]],
        colWidths=[5.9*cm]
    )
    gauge_cell.setStyle(TableStyle([
        ("ALIGN",   (0,0), (-1,-1), "CENTER"),
        ("VALIGN",  (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 0),
        ("BOTTOMPADDING", (0,0), (-1,-1), 0),
    ]))

    combined = Table([[metrics_table, gauge_cell]], colWidths=[11.7*cm, 5.7*cm])
    combined.setStyle(TableStyle([
        ("VALIGN",  (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
    ]))
    story.append(combined)
    story.append(Spacer(1, 0.5*cm))

    # ==========================================
    # SECTION 2: FITNESS SCORE PROGRESS BAR
    # ==========================================
    story.append(Paragraph("Fitness Score", section_style))
    story.append(HRFlowable(width=W, thickness=1.5, color=RED, spaceAfter=8))

    fitness_labels  = ["A - Excellent", "B - Good", "C - Fair", "D - Poor"]
    fitness_colors  = [
        colors.HexColor("#4caf50"),
        colors.HexColor("#2196f3"),
        colors.HexColor("#ff9800"),
        colors.HexColor("#f44336"),
    ]
    bar_fill = [1.0, 0.72, 0.44, 0.16]

    def make_fitness_progress(idx):
        d = Drawing(W, 70)
        bw = float(W) - 20
        bh = 18
        by = 35

        # Track
        d.add(Rect(10, by, bw, bh, rx=9, ry=9,
                   fillColor=colors.HexColor("#eeeeee"), strokeColor=None))
        # Fill
        fill_w = bw * bar_fill[idx]
        if fill_w > 0:
            d.add(Rect(10, by, fill_w, bh, rx=9, ry=9,
                       fillColor=fitness_colors[idx], strokeColor=None))

        # Grade markers
        for i, lbl in enumerate(["D", "C", "B", "A"]):
            mx = 10 + bw * bar_fill[3 - i]
            mc = fitness_colors[3 - i]
            d.add(Circle(mx, by + bh/2, 10,
                         fillColor=mc if (3 - i) >= idx else colors.HexColor("#cccccc"),
                         strokeColor=WHITE, strokeWidth=1.5))
            d.add(String(mx, by + bh/2 - 4, lbl, fontSize=8,
                         fillColor=WHITE, textAnchor='middle', fontName="Helvetica-Bold"))

        # Label
        label_txt = fitness_labels[idx]
        d.add(String(10, by + bh + 10, f"Category: {label_txt}",
                     fontSize=10, fillColor=fitness_colors[idx],
                     fontName="Helvetica-Bold"))
        return d

    story.append(make_fitness_progress(fitness_class_idx))
    story.append(Spacer(1, 0.3*cm))

    # ==========================================
    # SECTION 3: WEEKLY PLAN
    # ==========================================
    story.append(Paragraph("Weekly Training Plan", section_style))
    story.append(HRFlowable(width=W, thickness=1.5, color=RED, spaceAfter=6))

    en_days      = ["Sat","Sun","Mon","Tue","Wed","Thu","Fri"]
    en_workouts  = ["Push Day","Pull Day","Legs Day","Recovery","HIIT","Power","Rest"]
    en_nutrition = ["High Protein","High Protein","Carb Load","Vitamins","Low Carb","Fiber","Cheat Meal"]
    en_advice    = ["Hydrate","Stretch","8h Sleep","Walk","Focus","Intensity","Relax"]

    plan_data = [["Day", "Workout", "Nutrition", "Advice"]]
    for i in range(7):
        plan_data.append([en_days[i], en_workouts[i], en_nutrition[i], en_advice[i]])

    plan_table = Table(plan_data, colWidths=[2.8*cm, 5.2*cm, 5.2*cm, 4.2*cm])
    plan_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), BLUE),
        ("TEXTCOLOR",     (0,0), (-1,0), WHITE),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,0), 10),
        ("ALIGN",         (0,0), (-1,0), "CENTER"),
        ("TOPPADDING",    (0,0), (-1,0), 8),
        ("BOTTOMPADDING", (0,0), (-1,0), 8),
        ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",      (0,1), (-1,-1), 9),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,1), (-1,-1), 6),
        ("BOTTOMPADDING", (0,1), (-1,-1), 6),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [GRAY, WHITE]),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#dddddd")),
        ("FONTNAME",      (0,1), (0,-1), "Helvetica-Bold"),
        ("TEXTCOLOR",     (0,1), (0,-1), RED),
    ]))
    story.append(plan_table)
    story.append(Spacer(1, 0.5*cm))

    # ==========================================
    # SECTION 4: CONTACT + QR + LOGOS
    # ==========================================
    story.append(HRFlowable(width=W, thickness=1, color=colors.HexColor("#dddddd"), spaceAfter=8))
    story.append(Paragraph("Developer Contact", section_style))

    # --- QR Code ---
    qr_linkedin_url = "https://www.linkedin.com/in/k-ahmed-auc/"
    qr = qrcode.QRCode(version=1, box_size=4, border=2,
                       error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr.add_data(qr_linkedin_url)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="#1a237e", back_color="white")
    qr_buffer = io.BytesIO()
    qr_img.save(qr_buffer, format="PNG")
    qr_buffer.seek(0)
    qr_rl = RLImage(qr_buffer, width=2.5*cm, height=2.5*cm)

    # --- LinkedIn SVG كـ Drawing ---
    def make_linkedin_logo(size=22):
        d = Drawing(size, size)
        d.add(Rect(0, 0, size, size, rx=4, ry=4,
                   fillColor=colors.HexColor("#0077b5"), strokeColor=None))
        # "in" بشكل مبسط
        d.add(Rect(3, 3, 4, 4, fillColor=WHITE, strokeColor=None))
        d.add(Rect(3, 9, 4, 11, fillColor=WHITE, strokeColor=None))
        d.add(Rect(9, 9, 4, 11, fillColor=WHITE, strokeColor=None))
        d.add(Rect(9, 9, 9, 4,  fillColor=WHITE, strokeColor=None))
        d.add(Rect(13, 13, 4, 7, fillColor=WHITE, strokeColor=None))
        return d

    def make_github_logo(size=22):
        d = Drawing(size, size)
        d.add(Circle(size/2, size/2, size/2,
                     fillColor=colors.HexColor("#24292e"), strokeColor=None))
        # "G" مبسط
        d.add(Circle(size/2, size/2, size/2 - 3,
                     fillColor=colors.HexColor("#24292e"), strokeColor=WHITE, strokeWidth=1))
        d.add(String(size/2, size/2 - 4, "GH", fontSize=7,
                     fillColor=WHITE, textAnchor='middle', fontName="Helvetica-Bold"))
        return d

    linkedin_logo = make_linkedin_logo(24)
    github_logo   = make_github_logo(24)

    linkedin_label_style = ParagraphStyle(
        "LI", fontSize=9, textColor=colors.HexColor("#0077b5"),
        fontName="Helvetica-Bold", spaceBefore=2
    )
    github_label_style = ParagraphStyle(
        "GH", fontSize=9, textColor=colors.HexColor("#24292e"),
        fontName="Helvetica-Bold", spaceBefore=2
    )
    name_style = ParagraphStyle(
        "Name", fontSize=11, textColor=BLUE,
        fontName="Helvetica-Bold", spaceAfter=4
    )
    qr_note_style = ParagraphStyle(
        "QRNote", fontSize=7.5, textColor=colors.gray,
        alignment=TA_CENTER, fontName="Helvetica"
    )

    contact_info = Table([
        [Paragraph("Ahmed Khaled Gamal", name_style), ""],
        [linkedin_logo, Paragraph("linkedin.com/in/k-ahmed-auc/", linkedin_label_style)],
        [github_logo,   Paragraph("github.com/ahmedk-gamal",      github_label_style)],
    ], colWidths=[1.2*cm, 9*cm])
    contact_info.setStyle(TableStyle([
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING",   (0,0), (-1,-1), 4),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("SPAN",          (0,0), (-1, 0)),
    ]))

    qr_section = Table([
        [qr_rl],
        [Paragraph("Scan to connect\non LinkedIn", qr_note_style)]
    ], colWidths=[3*cm])
    qr_section.setStyle(TableStyle([
        ("ALIGN",   (0,0), (-1,-1), "CENTER"),
        ("VALIGN",  (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
    ]))

    contact_row = Table(
        [[contact_info, qr_section]],
        colWidths=[13*cm, 4.4*cm]
    )
    contact_row.setStyle(TableStyle([
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("BOX",          (0,0), (-1,-1), 0.5, colors.HexColor("#e0e0e0")),
        ("ROUNDEDCORNERS",(0,0),(-1,-1), [6,6,6,6]),
        ("BACKGROUND",   (0,0), (-1,-1), LBLUE),
        ("TOPPADDING",   (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0), (-1,-1), 8),
        ("LEFTPADDING",  (0,0), (0, 0),  12),
    ]))
    story.append(contact_row)
    story.append(Spacer(1, 0.4*cm))

    # ==========================================
    # FOOTER
    # ==========================================
    footer_divider = Table([[""]], colWidths=[W], rowHeights=[2])
    footer_divider.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), RED)]))
    story.append(footer_divider)
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        f"© 2026 Ahmed Khaled Gamal  |  Health AI Dashboard  |  All Rights Reserved  |  {now_str}",
        footer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# ==========================================
# 4. الدوال المساعدة
# ==========================================
def load_lottie(url):
    try: return requests.get(url).json()
    except: return None

@st.cache_resource
def load_ai_models():
    try:
        m = load_model('body_performance_model.keras')
        s = joblib.load('scaler.pkl')
        return m, s
    except: return None, None

heart_anim = load_lottie("https://assets5.lottiefiles.com/packages/lf20_m6cu96.json")
model, scaler = load_ai_models()

# ==========================================
# 5. قاموس اللغات
# ==========================================
texts = {
    "English": {
        "title": "Health AI Dashboard",
        "brief_h": "🤖 Technical Brief",
        "brief_txt": "This system utilizes a <b>Deep Learning MLP</b> model to classify physical performance. It features custom <b>Feature Engineering</b> (BMI & Pulse Pressure).",
        "dev_h": "👨‍💻 Developer Information",
        "btn": "Analyze & Generate Report ✨",
        "pdf_btn": "📥 Download PDF Report",
        "labels": ["Age","Gender","Height (cm)","Weight (kg)","Body Fat %",
                   "Systolic BP","Diastolic BP","Grip Force","Sit-ups","Jump (cm)","Flexibility"],
        "res_h": "📊 Comprehensive Results",
        "cat": "Fitness Category",
        "bmi_lab": "BMI Score",
        "pp_lab": "Pulse Pressure",
        "classes": ['A (Excellent)', 'B (Good)', 'C (Fair)', 'D (Poor)'],
        "days":      ["Sat","Sun","Mon","Tue","Wed","Thu","Fri"],
        "workouts":  ["Push Day","Pull Day","Legs Day","Recovery","HIIT","Power","Rest"],
        "nutrition": ["High Protein","High Protein","Carb Load","Vitamins","Low Carb","Fiber","Cheat Meal"],
        "advice":    ["Hydrate","Stretch","8h Sleep","Walk","Focus","Intensity","Relax"]
    },
    "العربية": {
        "title": "لوحة تحكم الأداء البدني الذكية",
        "brief_h": "🤖 نبذة تقنية",
        "brief_txt": "يعتمد النظام على نموذج <b>التعلم العميق (MLP)</b> للتنبؤ بمستوى اللياقة. تم دمج <b>هندسة البيانات</b> لحساب مؤشر كتلة الجسم وضغط النبض.",
        "dev_h": "👨‍💻 المطور",
        "btn": "تشغيل التحليل واستخراج التقرير ✨",
        "pdf_btn": "📥 تحميل التقرير الرسمي (PDF)",
        "labels": ["العمر","الجنس","الطول (سم)","الوزن (كجم)","نسبة الدهون %",
                   "الضغط الانقباضي","الضغط الانبساطي","قوة القبضة","تمارين البطن","الوثب","المرونة"],
        "res_h": "📊 النتائج والتحليلات",
        "cat": "تصنيف اللياقة",
        "bmi_lab": "مؤشر الكتلة",
        "pp_lab": "ضغط النبض",
        "classes": ['A (ممتاز)', 'B (جيد جداً)', 'C (متوسط)', 'D (ضعيف)'],
        "days":      ["السبت","الأحد","الاثنين","الثلاثاء","الأربعاء","الخميس","الجمعة"],
        "workouts":  ["تمارين دفع","تمارين سحب","تمارين أرجل","استشفاء","كارديو HIIT","تمارين قوة","راحة"],
        "nutrition": ["بروتين عالي","بروتين عالي","تحميل كارب","فيتامينات","كارب منخفض","ألياف","وجبة مفتوحة"],
        "advice":    ["شرب ماء","إطالات","نوم كافٍ","مشي","تركيز","كثافة","استشفاء"]
    }
}

# ==========================================
# 6. السايد بار
# ==========================================
with st.sidebar:
    st.markdown('<div class="sidebar-header"><div class="logo-text">HEALTH AI</div></div>', unsafe_allow_html=True)
    lang = st.radio("🌐 Language / اللغة", ("English", "العربية"))
    t = texts[lang]
    st.markdown("---")
    st.subheader(t["brief_h"])
    st.markdown(f'<div class="about-box">{t["brief_txt"]}</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.subheader(t["dev_h"])
    st.markdown("**Ahmed Khaled Gamal**")
    st.markdown("""
        <a href="https://www.linkedin.com/in/k-ahmed-auc/" class="social-link linkedin" target="_blank">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="#0077b5">
                <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 
                0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 
                0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 
                1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 
                0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>
            </svg>
            LinkedIn
        </a>
        <a href="https://github.com/ahmedk-gamal" class="social-link github" target="_blank">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="#333">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
            </svg>
            GitHub
        </a>
    """, unsafe_allow_html=True)

# ==========================================
# 7. المحتوى الرئيسي
# ==========================================
st.markdown('<div class="main-header-container">', unsafe_allow_html=True)
col_l, col_m, col_r = st.columns([1, 1, 1])
with col_m:
    if heart_anim:
        st_lottie(heart_anim, height=200, key="main_heart")
st.markdown(f'<div class="main-title-text">🧬 {t["title"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

if model is None:
    st.error("Model files not found!")
    st.stop()

with st.form("health_form"):
    c1, c2 = st.columns(2)
    with c1:
        age    = st.number_input(t["labels"][0], 10, 90, 25)
        gender = st.selectbox(t["labels"][1], ["Male","Female"] if lang=="English" else ["ذكر","أنثى"])
        h      = st.number_input(t["labels"][2], 120.0, 220.0, 175.0)
        w      = st.number_input(t["labels"][3], 30.0, 200.0, 75.0)
        f      = st.number_input(t["labels"][4], 5.0, 50.0, 20.0)
    with c2:
        s_bp = st.number_input(t["labels"][5], 80, 200, 120)
        d_bp = st.number_input(t["labels"][6], 50, 120, 80)
        grp  = st.number_input(t["labels"][7], 10.0, 100.0, 45.0)
        stp  = st.number_input(t["labels"][8], 0, 100, 35)
        jmp  = st.number_input(t["labels"][9], 50.0, 350.0, 210.0)
        bnd  = st.number_input(t["labels"][10], -20.0, 50.0, 15.0)
    run_btn = st.form_submit_button(t["btn"])

# ==========================================
# 8. النتائج والتقرير
# ==========================================
if run_btn:
    g_val = 0 if (gender in ["Male", "ذكر"]) else 1
    bmi   = w / ((h / 100) ** 2)
    pp    = s_bp - d_bp

    features = scaler.transform([[age, g_val, h, w, f, s_bp, d_bp, grp, stp, bnd, jmp, bmi, pp]])
    pred_idx = np.argmax(model.predict(features))

    st.markdown("---")
    st.subheader(t["res_h"])
    m1, m2, m3 = st.columns(3)
    m1.metric(t["cat"],     t["classes"][pred_idx])
    m2.metric(t["bmi_lab"], f"{bmi:.2f}")
    m3.metric(t["pp_lab"],  pp)

    plan_df = pd.DataFrame({
        "Day / اليوم":         t["days"],
        "Workout / التمرين":   t["workouts"],
        "Nutrition / التغذية": t["nutrition"],
        "Advice / نصيحة":      t["advice"]
    })
    st.table(plan_df)

    pdf_summary = {
        "Age":            age,
        "BMI Score":      f"{bmi:.2f}",
        "Pulse Pressure": pp,
        "Fitness":        t["classes"][pred_idx]
    }

    with st.spinner("Generating PDF..."):
        pdf_bytes = create_pdf(pdf_summary, plan_df, fitness_class_idx=pred_idx)

    st.download_button(
        label=t["pdf_btn"],
        data=pdf_bytes,
        file_name="Health_Report.pdf",
        mime="application/pdf"
    )

st.markdown("---")
st.caption("© 2026 Ahmed Khaled Gamal | All Rights Reserved")
