import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
from multilingual_voice import get_voice_report, get_language_html
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="FedHealth-Twin | Multi-Disease",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

.hero-banner {
    background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
    border-radius: 20px; padding: 28px;
    text-align: center; margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.hero-title {
    font-size: 2.8rem; font-weight: 700;
    background: linear-gradient(90deg, #ff4b6e, #1a73e8, #00c9ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 8px 0;
}
.hero-subtitle { color:#999; font-size:1rem; margin:4px 0; }
.badge {
    display:inline-block; padding:5px 15px; border-radius:20px;
    font-size:0.78rem; font-weight:600; margin:3px;
}
.badge-blue  { background:rgba(26,115,232,0.15); color:#4d9fff; border:1px solid #1a73e8; }
.badge-red   { background:rgba(255,75,110,0.15); color:#ff7090; border:1px solid #ff4b6e; }
.badge-green { background:rgba(0,201,87,0.15);   color:#00e676; border:1px solid #00c957; }
.badge-purple{ background:rgba(156,39,176,0.15); color:#ce93d8; border:1px solid #9c27b0; }
.badge-orange{ background:rgba(255,152,0,0.15);  color:#ffcc02; border:1px solid #ff9800; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#1a1a2e 0%,#16213e 100%);
}
.section-header {
    background:linear-gradient(90deg,rgba(26,115,232,0.2),transparent);
    border-left:4px solid #1a73e8; padding:10px 16px;
    border-radius:0 10px 10px 0; margin:20px 0 14px 0;
    color:white; font-size:1.1rem; font-weight:600;
}
.disease-card-safe {
    background: linear-gradient(135deg,#1b5e20,#2e7d32);
    border-radius:15px; padding:20px; text-align:center;
    color:white; border:1px solid #4caf50;
    box-shadow:0 4px 15px rgba(46,125,50,0.3); margin:5px;
}
.disease-card-risk {
    background: linear-gradient(135deg,#b71c1c,#c62828);
    border-radius:15px; padding:20px; text-align:center;
    color:white; border:1px solid #f44336;
    box-shadow:0 4px 15px rgba(198,40,40,0.4);
    margin:5px; animation:pulse 1.5s infinite;
}
@keyframes pulse {
    0%  { box-shadow:0 4px 15px rgba(198,40,40,0.3); }
    50% { box-shadow:0 4px 30px rgba(198,40,40,0.8); }
    100%{ box-shadow:0 4px 15px rgba(198,40,40,0.3); }
}
.disease-icon  { font-size:2rem; margin-bottom:6px; }
.disease-name  { font-size:1rem; font-weight:700; margin:4px 0; }
.disease-prob  { font-size:1.4rem; font-weight:700; margin:4px 0; }
.disease-status{ font-size:0.8rem; opacity:0.9; }
.info-card {
    background:rgba(255,255,255,0.05); border-radius:12px; padding:16px;
    border:1px solid rgba(255,255,255,0.1); text-align:center;
}
.info-card h3 { color:#1a73e8; margin:0; font-size:1.4rem; }
.info-card p  { color:#aaa; margin:5px 0 0 0; font-size:0.8rem; }
.vital-card {
    background:rgba(255,255,255,0.05); border-radius:12px; padding:14px 10px;
    border:1px solid rgba(255,255,255,0.1); text-align:center; margin:4px;
}
.vital-label  { color:#aaa; font-size:0.75rem; margin-bottom:4px; }
.vital-value  { font-size:1.4rem; font-weight:700; color:white; margin:2px 0; }
.vital-normal { font-size:0.7rem; margin-top:4px; font-weight:600; }
.vital-ok     { color:#00e676; }
.vital-warn   { color:#ffab40; }
.vital-danger { color:#ff5252; }
.stButton > button {
    background:linear-gradient(135deg,#ff4b6e,#1a73e8) !important;
    color:white !important; border:none !important;
    border-radius:25px !important; padding:12px 30px !important;
    font-weight:600 !important; width:100% !important;
    box-shadow:0 4px 15px rgba(255,75,110,0.4) !important;
}
.stDownloadButton > button {
    background:linear-gradient(135deg,#00c9ff,#0072ff) !important;
    color:white !important; border-radius:25px !important;
    border:none !important; font-weight:600 !important;
}
#MainMenu{visibility:hidden;} footer{visibility:hidden;} header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Load Models ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open('models/multidisease_models.pkl','rb') as f:
        models = pickle.load(f)
    with open('models/multidisease_scaler.pkl','rb') as f:
        scaler = pickle.load(f)
    return models, scaler

try:
    disease_models, multi_scaler = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    model_error   = str(e)


# ─── Vital Card Helper ────────────────────────────────────────
def vital_card(label, value, unit, low, high, normal_text, higher_bad=True):
    if higher_bad:
        if   value > high:       css, icon = "vital-danger","🔴"
        elif value > high*0.9:   css, icon = "vital-warn",  "🟡"
        else:                    css, icon = "vital-ok",    "🟢"
    else:
        if   value < low:        css, icon = "vital-danger","🔴"
        elif value < low*1.03:   css, icon = "vital-warn",  "🟡"
        else:                    css, icon = "vital-ok",    "🟢"
    return f"""
    <div class="vital-card">
        <div class="vital-label">{label}</div>
        <div class="vital-value">{value}{unit}</div>
        <div class="vital-normal {css}">{icon} {normal_text}</div>
    </div>"""


# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 14px 0;
         border-bottom:1px solid rgba(255,255,255,0.08);margin-bottom:14px;">
        <div style="font-size:2.8rem;">❤️</div>
        <h2 style="color:#ff4b6e;margin:4px 0 2px 0;font-size:1.3rem;">FedHealth-Twin</h2>
        <p style="color:#777;font-size:0.72rem;">Multi-Disease AI Prediction</p>
        <p style="color:#777;font-size:0.72rem;">🔒 Privacy Preserved | 🧠 AI Powered</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 👤 Demographics")
    patient_name = st.text_input("Patient Name", value="John Doe")
    age    = st.slider("Age (years)", 18, 90, 45)
    gender = st.selectbox("Gender", ["Male","Female"])

    st.markdown("### 🩺 Clinical Vitals")
    bmi            = st.slider("BMI",                   15.0, 50.0, 25.0, 0.1)
    blood_pressure = st.slider("Blood Pressure (mmHg)",  60,  200,  120)
    blood_sugar    = st.slider("Blood Sugar (mg/dL)",    70,  500,  100)
    cholesterol    = st.slider("Cholesterol (mg/dL)",   100,  400,  180)
    heart_rate     = st.slider("Heart Rate (bpm)",       40,  150,   75)
    spo2           = st.slider("SpO2 (%)",              80.0,100.0, 98.0, 0.1)
    creatinine     = st.slider("Creatinine (mg/dL)",    0.5,  6.0,  1.0, 0.1)

    st.markdown("### 🔬 Advanced Labs")
    hba1c         = st.slider("HbA1c (%)",             4.0, 12.0,  5.5, 0.1)
    triglycerides = st.slider("Triglycerides (mg/dL)",  50,  500,  150)
    alt_enzyme    = st.slider("ALT Enzyme (U/L)",       10,  200,   30)

    st.markdown("### 🏥 Medical History")
    smoker           = st.checkbox("🚬 Smoker")
    diabetes_history = st.checkbox("🩸 Diabetes History")
    family_history   = st.checkbox("👨‍👩‍👧 Family History")
    alcohol          = st.checkbox("🍺 Alcohol Consumption")
    physical_activity= st.selectbox("Physical Activity",
                                     ["Sedentary (0)","Moderate (1)","Active (2)"])
    st.markdown("---")
    predict_btn = st.button("🔍 Predict All Diseases", use_container_width=True)
    st.markdown("""
    <div style="text-align:center;margin-top:14px;color:#555;font-size:0.7rem;">
        🔒 Data never leaves your device<br>Powered by Federated Learning
    </div>""", unsafe_allow_html=True)


# ─── Hero ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div style="display:flex;justify-content:center;gap:18px;margin-bottom:12px;flex-wrap:wrap;">
    <div style="width:55px;height:55px;border-radius:50%;
         background:rgba(255,75,110,0.18);border:1.5px solid #ff4b6e;
         display:inline-flex;align-items:center;justify-content:center;font-size:1.6rem;">🩸</div>
    <div style="width:55px;height:55px;border-radius:50%;
         background:rgba(255,75,110,0.18);border:1.5px solid #ff4b6e;
         display:inline-flex;align-items:center;justify-content:center;font-size:1.6rem;">❤️</div>
    <div style="width:55px;height:55px;border-radius:50%;
         background:rgba(26,115,232,0.18);border:1.5px solid #1a73e8;
         display:inline-flex;align-items:center;justify-content:center;font-size:1.6rem;">🫘</div>
    <div style="width:55px;height:55px;border-radius:50%;
         background:rgba(0,201,87,0.18);border:1.5px solid #00c957;
         display:inline-flex;align-items:center;justify-content:center;font-size:1.6rem;">🩺</div>
    <div style="width:55px;height:55px;border-radius:50%;
         background:rgba(156,39,176,0.18);border:1.5px solid #9c27b0;
         display:inline-flex;align-items:center;justify-content:center;font-size:1.6rem;">🫀</div>
    <div style="width:55px;height:55px;border-radius:50%;
         background:rgba(255,152,0,0.18);border:1.5px solid #ff9800;
         display:inline-flex;align-items:center;justify-content:center;font-size:1.6rem;">🎙️</div>
  </div>
  <div class="hero-title">🏥 FedHealth-Twin</div>
  <div class="hero-subtitle">Multi-Disease AI Prediction System with Multilingual Voice</div>
  <div style="margin-top:12px;">
    <span class="badge badge-red">🩸 Diabetes</span>
    <span class="badge badge-red">❤️ Cardiovascular</span>
    <span class="badge badge-blue">🫘 Kidney (CKD)</span>
    <span class="badge badge-green">🩺 Hypertension</span>
    <span class="badge badge-purple">🫀 Fatty Liver</span>
    <span class="badge badge-orange">🎙️ EN | தமிழ் | हिंदी</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Prediction ───────────────────────────────────────────────
if predict_btn:
    if not models_loaded:
        st.error(f"❌ Models not loaded: {model_error}")
        st.stop()

    pa = int(physical_activity.split('(')[1].replace(')',''))

    patient_data = [age, 1 if gender=="Male" else 0, bmi,
                    blood_pressure, blood_sugar, cholesterol,
                    heart_rate, spo2, creatinine, hba1c,
                    triglycerides, alt_enzyme,
                    1 if smoker else 0,
                    1 if diabetes_history else 0,
                    1 if family_history else 0,
                    1 if alcohol else 0, pa]

    feature_cols = ['age','gender','bmi','blood_pressure','blood_sugar',
                    'cholesterol','heart_rate','spo2','creatinine','hba1c',
                    'triglycerides','alt_enzyme','smoker','diabetes_history',
                    'family_history','alcohol','physical_activity']

    pdf        = pd.DataFrame([patient_data], columns=feature_cols)
    pdf_scaled = multi_scaler.transform(pdf)

    # Predict all 5 diseases
    disease_info = {
        'diabetes':     ('🩸','Diabetes',        '#e53935'),
        'cvd':          ('❤️','Cardiovascular',  '#e53935'),
        'ckd':          ('🫘','Kidney (CKD)',     '#1e88e5'),
        'hypertension': ('🩺','Hypertension',     '#43a047'),
        'fatty_liver':  ('🫀','Fatty Liver',      '#8e24aa'),
    }

    predictions   = {}
    probabilities = {}
    for disease, model in disease_models.items():
        pred = model.predict(pdf_scaled)[0]
        prob = model.predict_proba(pdf_scaled)[0][1] * 100
        predictions[disease]   = pred
        probabilities[disease] = prob

    risk_diseases = [d for d,p in predictions.items() if p == 1]
    safe_diseases = [d for d,p in predictions.items() if p == 0]

    # Build recommendations
    disease_recs = {
        'diabetes':     ["Monitor blood sugar daily",
                         "Reduce refined carbs and sweets",
                         "Exercise 30 min daily",
                         "Consult endocrinologist",
                         "Check HbA1c every 3 months"],
        'cvd':          ["Reduce saturated fat intake",
                         "Stop smoking immediately",
                         "30 min cardio exercise daily",
                         "Monitor cholesterol monthly",
                         "Consult cardiologist urgently"],
        'ckd':          ["Reduce protein and salt intake",
                         "Monitor creatinine weekly",
                         "Control blood pressure strictly",
                         "Avoid nephrotoxic drugs",
                         "Consult nephrologist immediately"],
        'hypertension': ["Reduce salt intake drastically",
                         "Exercise daily",
                         "Monitor BP twice daily",
                         "Avoid stress and alcohol",
                         "Consult physician for medication"],
        'fatty_liver':  ["Avoid alcohol completely",
                         "Lose weight gradually",
                         "Reduce fatty and fried foods",
                         "Exercise 5 days per week",
                         "Get liver function test monthly"],
    }

    all_recs = []
    for disease in risk_diseases:
        icon, name, _ = disease_info[disease]
        all_recs.extend([f"{name}: {r}" for r in disease_recs[disease]])
    if not all_recs:
        all_recs = ["Maintain regular exercise",
                    "Eat a balanced diet",
                    "Get health checkup every 6 months",
                    "Stay hydrated"]

    # ── Overall Summary ───────────────────────────────────────
    st.markdown('<div class="section-header">🎯 Overall Risk Summary</div>',
                unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f'<div class="info-card"><h3>{len(risk_diseases)}</h3><p>Diseases at Risk</p></div>',
                unsafe_allow_html=True)
    c2.markdown(f'<div class="info-card"><h3>{len(safe_diseases)}</h3><p>Diseases Safe</p></div>',
                unsafe_allow_html=True)
    c3.markdown(f'<div class="info-card"><h3>{len(risk_diseases)*20}%</h3><p>Overall Risk Score</p></div>',
                unsafe_allow_html=True)
    c4.markdown(f'<div class="info-card"><h3>{patient_name[:8]}</h3><p>Patient</p></div>',
                unsafe_allow_html=True)

    if len(risk_diseases) == 0:
        st.success("✅ Excellent! No disease risk detected. Maintain your healthy lifestyle!")
    elif len(risk_diseases) <= 2:
        st.warning(f"⚠️ Moderate Risk! {len(risk_diseases)} disease(s) need attention.")
    else:
        st.error(f"🚨 High Risk! {len(risk_diseases)} diseases detected. Seek immediate medical attention!")

    st.markdown("---")

    # ── Disease Cards ─────────────────────────────────────────
    st.markdown('<div class="section-header">🔬 Disease-by-Disease Analysis</div>',
                unsafe_allow_html=True)

    cols = st.columns(5)
    for col,(disease,(icon,name,color)) in zip(cols, disease_info.items()):
        prob   = probabilities[disease]
        pred   = predictions[disease]
        css    = "disease-card-risk" if pred==1 else "disease-card-safe"
        status = "⚠️ AT RISK" if pred==1 else "✅ SAFE"
        col.markdown(f"""
        <div class="{css}">
            <div class="disease-icon">{icon}</div>
            <div class="disease-name">{name}</div>
            <div class="disease-prob">{prob:.1f}%</div>
            <div class="disease-status">{status}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Disease Risk Analysis</div>',
                unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    colors_map = {'diabetes':'#e53935','cvd':'#e53935',
                  'ckd':'#1e88e5','hypertension':'#43a047','fatty_liver':'#8e24aa'}

    with c1:
        disease_names_list = ['Diabetes','CVD','Kidney','Hypertension','Fatty Liver']
        probs_list = [probabilities[d] for d in disease_info.keys()]
        angles = np.linspace(0, 2*np.pi, len(disease_names_list), endpoint=False).tolist()
        pp = probs_list + [probs_list[0]]
        aa = angles    + [angles[0]]

        fig,ax = plt.subplots(figsize=(5,5),
                               subplot_kw=dict(polar=True),
                               facecolor='#16213e')
        ax.set_facecolor('#16213e')
        ax.plot(aa, pp, 'o-', linewidth=2, color='#ff4b6e')
        ax.fill(aa, pp, alpha=0.25, color='#ff4b6e')
        ax.set_xticks(angles)
        ax.set_xticklabels(disease_names_list, color='white', fontsize=9)
        ax.set_ylim(0,100)
        ax.set_yticks([20,40,60,80,100])
        ax.set_yticklabels(['20','40','60','80','100'], color='#888', fontsize=7)
        ax.grid(color='#333', linewidth=0.8)
        ax.spines['polar'].set_color('#333')
        ax.set_title(f"Risk Radar — {patient_name}",
                     color='white', fontweight='bold', fontsize=11, pad=15)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c2:
        colors_bar = ['#e53935' if predictions[d]==1 else '#2e7d32'
                      for d in disease_info.keys()]
        fig,ax = plt.subplots(figsize=(5,5), facecolor='#16213e')
        ax.set_facecolor('#16213e')
        bars = ax.barh(disease_names_list, probs_list,
                       color=colors_bar, edgecolor='#ffffff11', height=0.5)
        for bar,val in zip(bars,probs_list):
            ax.text(min(val+1,95), bar.get_y()+bar.get_height()/2,
                    f'{val:.1f}%', va='center',
                    color='white', fontweight='bold', fontsize=9)
        ax.axvline(x=50, color='#ffcc02', linewidth=1.5,
                   linestyle='--', alpha=0.7, label='Risk Threshold (50%)')
        ax.set_xlim(0,110)
        ax.set_xlabel("Risk Probability (%)", color='#aaa', fontsize=9)
        ax.set_title("Disease Risk Levels", color='white',
                     fontweight='bold', fontsize=11, pad=10)
        ax.tick_params(colors='white', labelsize=9)
        for sp in ['top','right']:   ax.spines[sp].set_visible(False)
        for sp in ['bottom','left']: ax.spines[sp].set_color('#333')
        ax.legend(loc='lower right', fontsize=8,
                  facecolor='#1a1a2e', labelcolor='white')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")

    # ── Vitals Summary ────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Patient Vitals Summary</div>',
                unsafe_allow_html=True)

    r1 = st.columns(4)
    for col,args in zip(r1,[
        ("BMI",            bmi,            "",      18.5, 25,  "Normal: 18.5–25",  True),
        ("Blood Pressure", blood_pressure, " mmHg",  60, 120,  "Normal: <120",     True),
        ("Blood Sugar",    blood_sugar,   " mg/dL",  70, 140,  "Normal: 70–140",   True),
        ("SpO2",           spo2,          "%",       95, 100,  "Normal: >95%",     False),
    ]):
        col.markdown(vital_card(*args), unsafe_allow_html=True)

    r2 = st.columns(4)
    for col,args in zip(r2,[
        ("HbA1c",        hba1c,         "%",     4.0, 5.7, "Normal: <5.7%",    True),
        ("Triglycerides",triglycerides, " mg/dL", 50, 150, "Normal: <150",      True),
        ("ALT Enzyme",   alt_enzyme,    " U/L",   10,  40, "Normal: 10–40",     True),
        ("Creatinine",   creatinine,    " mg/dL", 0.5, 1.2,"Normal: 0.5–1.2",  True),
    ]):
        col.markdown(vital_card(*args), unsafe_allow_html=True)

    st.markdown("---")

    # ── Recommendations ───────────────────────────────────────
    st.markdown('<div class="section-header">💡 Disease-Specific Recommendations</div>',
                unsafe_allow_html=True)

    if risk_diseases:
        for disease in risk_diseases:
            icon,name,_ = disease_info[disease]
            with st.expander(f"{icon} {name} — AT RISK ⚠️", expanded=True):
                for i,rec in enumerate(disease_recs[disease],1):
                    st.markdown(f"**{i}.** {rec}")
    else:
        st.success("✅ All diseases are in safe range! Keep up the healthy lifestyle.")

    st.markdown("---")

    # ── Multilingual Voice Assistant ──────────────────────────
    st.markdown('<div class="section-header">🎙️ Multilingual Voice Assistant — EN | தமிழ் | हिंदी</div>',
                unsafe_allow_html=True)

    en_text = get_voice_report(
        patient_name, age, gender, risk_name="",
        confidence=0, probabilities=probabilities,
        predictions=predictions, recs=all_recs, language='en'
    )
    ta_text = get_voice_report(
        patient_name, age, gender, risk_name="",
        confidence=0, probabilities=probabilities,
        predictions=predictions, recs=all_recs, language='ta'
    )
    hi_text = get_voice_report(
        patient_name, age, gender, risk_name="",
        confidence=0, probabilities=probabilities,
        predictions=predictions, recs=all_recs, language='hi'
    )

    st.components.v1.html(
        get_language_html(en_text, ta_text, hi_text),
        height=270
    )

    st.markdown("---")

    # ── Report Download ───────────────────────────────────────
    st.markdown('<div class="section-header">📄 Download Health Report</div>',
                unsafe_allow_html=True)

    now        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    risk_names_str = ", ".join([disease_info[d][1] for d in risk_diseases]) \
                     if risk_diseases else "None"

    report = f"""
FedHealth-Twin: Multi-Disease Health Report
Generated : {now}
Patient   : {patient_name}
{'='*60}
PATIENT DETAILS:
  Age            : {age} years
  Gender         : {gender}
  BMI            : {bmi}
  Blood Pressure : {blood_pressure} mmHg
  Blood Sugar    : {blood_sugar} mg/dL
  HbA1c          : {hba1c}%
  Cholesterol    : {cholesterol} mg/dL
  Triglycerides  : {triglycerides} mg/dL
  ALT Enzyme     : {alt_enzyme} U/L
  Heart Rate     : {heart_rate} bpm
  SpO2           : {spo2}%
  Creatinine     : {creatinine} mg/dL
  Smoker         : {'Yes' if smoker else 'No'}
  Diabetes Hx    : {'Yes' if diabetes_history else 'No'}
  Family Hx      : {'Yes' if family_history else 'No'}
  Alcohol        : {'Yes' if alcohol else 'No'}
{'='*60}
DISEASE PREDICTIONS:
  Diabetes       : {'AT RISK ⚠️' if predictions['diabetes']     else 'SAFE ✅'} ({probabilities['diabetes']:.1f}%)
  Cardiovascular : {'AT RISK ⚠️' if predictions['cvd']          else 'SAFE ✅'} ({probabilities['cvd']:.1f}%)
  Kidney (CKD)   : {'AT RISK ⚠️' if predictions['ckd']          else 'SAFE ✅'} ({probabilities['ckd']:.1f}%)
  Hypertension   : {'AT RISK ⚠️' if predictions['hypertension'] else 'SAFE ✅'} ({probabilities['hypertension']:.1f}%)
  Fatty Liver    : {'AT RISK ⚠️' if predictions['fatty_liver']  else 'SAFE ✅'} ({probabilities['fatty_liver']:.1f}%)
{'='*60}
DISEASES AT RISK : {risk_names_str}
OVERALL STATUS   : {len(risk_diseases)}/5 diseases at risk
{'='*60}
RECOMMENDATIONS:
""" + "\n".join([f"  - {r}" for r in all_recs]) + \
f"""
{'='*60}
DISCLAIMER: This report is AI-generated for informational
purposes only. Always consult a qualified medical professional.
{'='*60}
"""
    st.text_area("Full Report", report, height=280)
    st.download_button(
        "⬇️ Download Multi-Disease Report",
        data=report,
        file_name=f"{patient_name}_multidisease_report.txt",
        mime="text/plain"
    )

    st.markdown("---")

    # ── Privacy ───────────────────────────────────────────────
    st.markdown('<div class="section-header">🔒 Federated Learning Privacy</div>',
                unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    for col,name in zip([c1,c2,c3],
                        ["🏥 Hospital A","🏥 Hospital B","🏥 Hospital C"]):
        col.markdown(f'<div class="info-card"><h3>{name}</h3>'
                     f'<p>Local Training Only<br>Data stays private</p></div>',
                     unsafe_allow_html=True)
    st.success("🔒 Only model weights shared — Patient data NEVER leaves local systems!")


# ─── Welcome Screen ──────────────────────────────────────────
else:
    st.markdown("""
    <div style="text-align:center;padding:30px;color:#aaa;">
        <div style="font-size:3rem;margin-bottom:15px;">🔬</div>
        <h2 style="color:white;">Multi-Disease AI Prediction</h2>
        <p>Enter patient details in the sidebar and click<br>
        <strong style="color:#ff4b6e;">🔍 Predict All Diseases</strong>
        to get a complete 5-disease health analysis<br>
        with <strong style="color:#ffcc02;">🎙️ Voice in English, Tamil & Hindi</strong></p>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(icon,name,desc) in zip([c1,c2,c3,c4,c5],[
        ("🩸","Diabetes",      "Blood sugar & HbA1c"),
        ("❤️","Cardiovascular","Heart & cholesterol"),
        ("🫘","Kidney (CKD)",  "Creatinine & BP"),
        ("🩺","Hypertension",  "Blood pressure risk"),
        ("🫀","Fatty Liver",   "Triglycerides & ALT"),
    ]):
        col.markdown(f"""
        <div style="background:rgba(255,255,255,0.05);border-radius:15px;
             padding:20px;text-align:center;
             border:1px solid rgba(255,255,255,0.1);">
            <div style="font-size:2rem;">{icon}</div>
            <div style="color:white;font-weight:600;
                 font-size:0.9rem;margin-top:6px;">{name}</div>
            <div style="color:#777;font-size:0.75rem;
                 margin-top:4px;">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    c1,c2,c3 = st.columns(3)
    for col,(icon,title,desc) in zip([c1,c2,c3],[
        ("🎙️","Multilingual Voice","Speak in English, Tamil & Hindi"),
        ("📊","Radar Chart","Visual disease risk analysis"),
        ("🔒","Federated Privacy","Data never leaves your device"),
    ]):
        col.markdown(f"""
        <div style="background:rgba(255,255,255,0.05);border-radius:15px;
             padding:20px;text-align:center;
             border:1px solid rgba(255,255,255,0.1);">
            <div style="font-size:2rem;">{icon}</div>
            <div style="color:white;font-weight:600;
                 font-size:0.9rem;margin-top:6px;">{title}</div>
            <div style="color:#777;font-size:0.75rem;
                 margin-top:4px;">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#444;padding:16px;font-size:0.85rem;">
        ❤️ FedHealth-Twin | Multi-Disease AI Prediction System
    </div>""", unsafe_allow_html=True)
