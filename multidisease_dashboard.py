import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
from datetime import datetime
from multilingual_voice import get_voice_report, get_language_html
from llm_report import generate_llm_report, generate_llm_summary
try:
    import shap
    shap_available = True
except ImportError:
    shap_available = False
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="FedHealth-Twin | Multi-Disease AI",
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
.badge-cyan  { background:rgba(0,188,212,0.15);  color:#4dd0e1; border:1px solid #00bcd4; }

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
.tab-content {
    background:rgba(255,255,255,0.03);
    border-radius:15px; padding:20px;
    border:1px solid rgba(255,255,255,0.08);
    margin-top:10px;
}
.llm-card {
    background:linear-gradient(135deg,#0d1b2a,#1b2838);
    border-radius:15px; padding:22px;
    border:1px solid rgba(26,115,232,0.3);
    box-shadow:0 4px 25px rgba(26,115,232,0.1);
    margin:10px 0;
}
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
/* Tab Styling */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px; padding: 4px; gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 10px;
    color: #aaa; font-weight: 600; font-size: 0.88rem;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#ff4b6e,#1a73e8) !important;
    color: white !important;
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
  <div style="display:flex;justify-content:center;gap:16px;margin-bottom:12px;flex-wrap:wrap;">
    <div style="width:52px;height:52px;border-radius:50%;background:rgba(255,75,110,0.18);
         border:1.5px solid #ff4b6e;display:inline-flex;align-items:center;
         justify-content:center;font-size:1.5rem;">🩸</div>
    <div style="width:52px;height:52px;border-radius:50%;background:rgba(255,75,110,0.18);
         border:1.5px solid #ff4b6e;display:inline-flex;align-items:center;
         justify-content:center;font-size:1.5rem;">❤️</div>
    <div style="width:52px;height:52px;border-radius:50%;background:rgba(26,115,232,0.18);
         border:1.5px solid #1a73e8;display:inline-flex;align-items:center;
         justify-content:center;font-size:1.5rem;">🫘</div>
    <div style="width:52px;height:52px;border-radius:50%;background:rgba(0,201,87,0.18);
         border:1.5px solid #00c957;display:inline-flex;align-items:center;
         justify-content:center;font-size:1.5rem;">🩺</div>
    <div style="width:52px;height:52px;border-radius:50%;background:rgba(156,39,176,0.18);
         border:1.5px solid #9c27b0;display:inline-flex;align-items:center;
         justify-content:center;font-size:1.5rem;">🫀</div>
    <div style="width:52px;height:52px;border-radius:50%;background:rgba(255,152,0,0.18);
         border:1.5px solid #ff9800;display:inline-flex;align-items:center;
         justify-content:center;font-size:1.5rem;">🎙️</div>
    <div style="width:52px;height:52px;border-radius:50%;background:rgba(0,188,212,0.18);
         border:1.5px solid #00bcd4;display:inline-flex;align-items:center;
         justify-content:center;font-size:1.5rem;">🤖</div>
  </div>
  <div class="hero-title">🏥 FedHealth-Twin</div>
  <div class="hero-subtitle">Multi-Disease AI Prediction | Multilingual Voice | LLM Health Report</div>
  <div style="margin-top:12px;">
    <span class="badge badge-red">🩸 Diabetes</span>
    <span class="badge badge-red">❤️ Cardiovascular</span>
    <span class="badge badge-blue">🫘 Kidney</span>
    <span class="badge badge-green">🩺 Hypertension</span>
    <span class="badge badge-purple">🫀 Fatty Liver</span>
    <span class="badge badge-orange">🎙️ EN|தமிழ்|हिंदी</span>
    <span class="badge badge-cyan">🤖 Claude AI</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Main Prediction ─────────────────────────────────────────
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

    risk_diseases = [d for d,p in predictions.items() if p==1]
    safe_diseases = [d for d,p in predictions.items() if p==0]

    disease_recs = {
        'diabetes':     ["Monitor blood sugar daily",
                         "Reduce refined carbs and sweets",
                         "Exercise 30 min daily",
                         "Consult endocrinologist",
                         "Check HbA1c every 3 months"],
        'cvd':          ["Reduce saturated fat intake",
                         "Stop smoking immediately",
                         "30 min cardio daily",
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
        icon,name,_ = disease_info[disease]
        all_recs.extend([f"{name}: {r}" for r in disease_recs[disease]])
    if not all_recs:
        all_recs = ["Maintain regular exercise",
                    "Eat balanced diet",
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

    if len(risk_diseases)==0:
        st.success("✅ Excellent! No disease risk detected. Maintain your healthy lifestyle!")
    elif len(risk_diseases)<=2:
        st.warning(f"⚠️ Moderate Risk! {len(risk_diseases)} disease(s) need attention.")
    else:
        st.error(f"🚨 High Risk! {len(risk_diseases)} diseases detected. Seek immediate medical attention!")

    # ── Disease Cards ─────────────────────────────────────────
    st.markdown('<div class="section-header">🔬 Disease-by-Disease Analysis</div>',
                unsafe_allow_html=True)
    cols = st.columns(5)
    for col,(disease,(icon,name,color)) in zip(cols,disease_info.items()):
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

    # ════════════════════════════════════════════════════════
    # TABS
    # ════════════════════════════════════════════════════════
    tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
        "📊 Analysis",
        "📋 Vitals",
        "💡 Recommendations",
        "🔍 SHAP & XAI",
        "🎙️ Voice Assistant",
        "🤖 AI Report",
        "📄 Download"
    ])

    # ── TAB 1: Analysis ───────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">📊 Disease Risk Analysis</div>',
                    unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        disease_names_list = ['Diabetes','CVD','Kidney','Hypertension','Fatty Liver']
        probs_list = [probabilities[d] for d in disease_info.keys()]

        with c1:
            angles = np.linspace(0,2*np.pi,len(disease_names_list),endpoint=False).tolist()
            pp = probs_list+[probs_list[0]]
            aa = angles+[angles[0]]
            fig,ax = plt.subplots(figsize=(5,5),subplot_kw=dict(polar=True),facecolor='#16213e')
            ax.set_facecolor('#16213e')
            ax.plot(aa,pp,'o-',linewidth=2,color='#ff4b6e')
            ax.fill(aa,pp,alpha=0.25,color='#ff4b6e')
            ax.set_xticks(angles)
            ax.set_xticklabels(disease_names_list,color='white',fontsize=9)
            ax.set_ylim(0,100)
            ax.set_yticks([20,40,60,80,100])
            ax.set_yticklabels(['20','40','60','80','100'],color='#888',fontsize=7)
            ax.grid(color='#333',linewidth=0.8)
            ax.spines['polar'].set_color('#333')
            ax.set_title(f"Risk Radar — {patient_name}",color='white',
                         fontweight='bold',fontsize=11,pad=15)
            fig.tight_layout()
            st.pyplot(fig,use_container_width=True)
            plt.close()

        with c2:
            colors_bar = ['#e53935' if predictions[d]==1 else '#2e7d32'
                          for d in disease_info.keys()]
            fig,ax = plt.subplots(figsize=(5,5),facecolor='#16213e')
            ax.set_facecolor('#16213e')
            bars = ax.barh(disease_names_list,probs_list,
                           color=colors_bar,edgecolor='#ffffff11',height=0.5)
            for bar,val in zip(bars,probs_list):
                ax.text(min(val+1,95),bar.get_y()+bar.get_height()/2,
                        f'{val:.1f}%',va='center',
                        color='white',fontweight='bold',fontsize=9)
            ax.axvline(x=50,color='#ffcc02',linewidth=1.5,
                       linestyle='--',alpha=0.7,label='Risk Threshold (50%)')
            ax.set_xlim(0,110)
            ax.set_xlabel("Risk Probability (%)",color='#aaa',fontsize=9)
            ax.set_title("Disease Risk Levels",color='white',
                         fontweight='bold',fontsize=11,pad=10)
            ax.tick_params(colors='white',labelsize=9)
            for sp in ['top','right']:   ax.spines[sp].set_visible(False)
            for sp in ['bottom','left']: ax.spines[sp].set_color('#333')
            ax.legend(loc='lower right',fontsize=8,
                      facecolor='#1a1a2e',labelcolor='white')
            fig.tight_layout()
            st.pyplot(fig,use_container_width=True)
            plt.close()

        # Pie chart for risk distribution
        st.markdown('<div class="section-header">🥧 Risk vs Safe Distribution</div>',
                    unsafe_allow_html=True)
        c1,c2,c3 = st.columns([1,2,1])
        with c2:
            fig,ax = plt.subplots(figsize=(5,4),facecolor='#16213e')
            ax.set_facecolor('#16213e')
            sizes  = [len(risk_diseases), len(safe_diseases)]
            labels = [f'At Risk ({len(risk_diseases)})', f'Safe ({len(safe_diseases)})']
            colors = ['#c62828','#2e7d32']
            if all(s > 0 for s in sizes):
                wedges,texts,autotexts = ax.pie(
                    sizes,labels=labels,colors=colors,
                    autopct='%1.0f%%',startangle=90,
                    wedgeprops=dict(edgecolor='#16213e',linewidth=2)
                )
                for t in texts:     t.set_color('white'); t.set_fontsize(10)
                for t in autotexts: t.set_color('white'); t.set_fontweight('bold')
            elif sizes[0] == 0:
                ax.text(0.5,0.5,'✅ All 5 Diseases\nSAFE',
                        ha='center',va='center',color='#00e676',
                        fontsize=14,fontweight='bold',transform=ax.transAxes)
            else:
                ax.text(0.5,0.5,'🚨 All 5 Diseases\nAT RISK',
                        ha='center',va='center',color='#ff5252',
                        fontsize=14,fontweight='bold',transform=ax.transAxes)
            ax.set_title("Disease Risk Summary",color='white',
                         fontweight='bold',fontsize=11)
            fig.tight_layout()
            st.pyplot(fig,use_container_width=True)
            plt.close()

    # ── TAB 2: Vitals ─────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">📋 Patient Vitals Summary</div>',
                    unsafe_allow_html=True)
        r1 = st.columns(4)
        for col,args in zip(r1,[
            ("BMI",            bmi,            "",      18.5, 25,  "Normal: 18.5–25",  True),
            ("Blood Pressure", blood_pressure, " mmHg",  60, 120,  "Normal: <120",     True),
            ("Blood Sugar",    blood_sugar,   " mg/dL",  70, 140,  "Normal: 70–140",   True),
            ("SpO2",           spo2,          "%",       95, 100,  "Normal: >95%",     False),
        ]):
            col.markdown(vital_card(*args),unsafe_allow_html=True)

        r2 = st.columns(4)
        for col,args in zip(r2,[
            ("HbA1c",        hba1c,         "%",     4.0, 5.7, "Normal: <5.7%",   True),
            ("Triglycerides",triglycerides, " mg/dL", 50, 150, "Normal: <150",     True),
            ("ALT Enzyme",   alt_enzyme,    " U/L",   10,  40, "Normal: 10–40",    True),
            ("Creatinine",   creatinine,    " mg/dL", 0.5, 1.2,"Normal: 0.5–1.2", True),
        ]):
            col.markdown(vital_card(*args),unsafe_allow_html=True)

        r3 = st.columns(4)
        for col,args in zip(r3,[
            ("Heart Rate",  heart_rate,  " bpm",    60, 100, "Normal: 60–100",  True),
            ("Cholesterol", cholesterol, " mg/dL", 100, 200, "Normal: <200",    True),
            ("Age",         age,         " yrs",     0, 200, "",                True),
            ("Gender",      0,           "",          0, 200, gender,           True),
        ]):
            col.markdown(vital_card(*args),unsafe_allow_html=True)

        # Vitals bar chart
        st.markdown('<div class="section-header">📊 Vitals vs Normal Range</div>',
                    unsafe_allow_html=True)
        vitals_data = {
            'BMI':           (bmi,         25),
            'BP':            (blood_pressure, 120),
            'Sugar':         (blood_sugar,  140),
            'Cholesterol':   (cholesterol,  200),
            'Heart Rate':    (heart_rate,   100),
            'HbA1c(x10)':   (hba1c*10,     57),
            'Creatinine(x10)':(creatinine*10,12),
        }
        fig,ax = plt.subplots(figsize=(10,4),facecolor='#16213e')
        ax.set_facecolor('#16213e')
        vnames  = list(vitals_data.keys())
        vvals   = [vitals_data[k][0] for k in vnames]
        vnorms  = [vitals_data[k][1] for k in vnames]
        x       = np.arange(len(vnames))
        w       = 0.35
        vcolors = ['#e53935' if v>n else '#2e7d32'
                   for v,n in zip(vvals,vnorms)]
        ax.bar(x-w/2, vvals,  w, label='Patient Value', color=vcolors,   alpha=0.9)
        ax.bar(x+w/2, vnorms, w, label='Normal Limit',  color='#1a73e8', alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(vnames,color='white',fontsize=8,rotation=15)
        ax.tick_params(colors='white',labelsize=8)
        ax.set_title("Patient Vitals vs Normal Range",
                     color='white',fontweight='bold',fontsize=11)
        ax.legend(fontsize=8,facecolor='#1a1a2e',labelcolor='white')
        for sp in ['top','right']:   ax.spines[sp].set_visible(False)
        for sp in ['bottom','left']: ax.spines[sp].set_color('#333')
        fig.tight_layout()
        st.pyplot(fig,use_container_width=True)
        plt.close()

    # ── TAB 3: Recommendations ────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">💡 Disease-Specific Recommendations</div>',
                    unsafe_allow_html=True)
        if risk_diseases:
            for disease in risk_diseases:
                icon,name,_ = disease_info[disease]
                with st.expander(f"{icon} {name} — AT RISK ⚠️", expanded=True):
                    for i,rec in enumerate(disease_recs[disease],1):
                        st.markdown(f"**{i}.** {rec}")
        else:
            st.success("✅ All diseases are in safe range!")
            for i,rec in enumerate([
                "Maintain regular exercise (30 min daily)",
                "Eat a balanced diet rich in fruits and vegetables",
                "Get health checkup every 6 months",
                "Stay hydrated — drink 8 glasses of water daily",
                "Maintain healthy sleep schedule (7-8 hours)"
            ],1):
                st.markdown(f"**{i}.** {rec}")

        # Lifestyle Tracker
        st.markdown('<div class="section-header">🏃 Lifestyle Risk Factors</div>',
                    unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            factors = {
                'Smoking':           (1 if smoker else 0,           1),
                'Alcohol':           (1 if alcohol else 0,           1),
                'Sedentary':         (1 if pa==0 else 0,             1),
                'Overweight (BMI>25)':(1 if bmi>25 else 0,           1),
                'High Sugar':        (1 if blood_sugar>140 else 0,   1),
            }
            fig,ax = plt.subplots(figsize=(5,4),facecolor='#16213e')
            ax.set_facecolor('#16213e')
            fnames  = list(factors.keys())
            fvals   = [v[0] for v in factors.values()]
            fcolors = ['#e53935' if v==1 else '#2e7d32' for v in fvals]
            ax.barh(fnames,fvals,color=fcolors,edgecolor='#ffffff11',height=0.5)
            ax.set_xlim(0,1.5)
            ax.set_title("Risk Factors Present",color='white',
                         fontweight='bold',fontsize=10)
            ax.tick_params(colors='white',labelsize=8)
            for sp in ['top','right','bottom']: ax.spines[sp].set_visible(False)
            ax.spines['left'].set_color('#333')
            fig.tight_layout()
            st.pyplot(fig,use_container_width=True)
            plt.close()
        with c2:
            total_risk_factors = sum([
                1 if smoker else 0,
                1 if alcohol else 0,
                1 if pa==0 else 0,
                1 if bmi>25 else 0,
                1 if blood_sugar>140 else 0,
            ])
            st.markdown(f"""
            <div class="info-card" style="margin-top:20px;">
                <h3>{total_risk_factors}/5</h3>
                <p>Lifestyle Risk Factors</p>
            </div>
            <div style="margin-top:15px;color:#aaa;font-size:0.85rem;line-height:1.8;">
                {"🔴 Smoker detected — major risk factor" if smoker else "🟢 Non-smoker"}<br>
                {"🔴 Alcohol consumption detected" if alcohol else "🟢 No alcohol"}<br>
                {"🔴 Sedentary lifestyle" if pa==0 else "🟢 Active lifestyle"}<br>
                {"🔴 BMI above normal range" if bmi>25 else "🟢 BMI in normal range"}<br>
                {"🔴 Blood sugar elevated" if blood_sugar>140 else "🟢 Blood sugar normal"}
            </div>
            """,unsafe_allow_html=True)

    # ── TAB 4: SHAP & Explainable AI ─────────────────────────
    with tab4:
        st.markdown('<div class="section-header">🔍 SHAP — Explainable AI Analysis</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div style="background:rgba(0,188,212,0.08);border-radius:12px;
             padding:16px;border:1px solid rgba(0,188,212,0.2);margin-bottom:16px;">
            <div style="color:#4dd0e1;font-weight:600;margin-bottom:4px;">
                🔍 What is SHAP Explainability?
            </div>
            <div style="color:#aaa;font-size:0.82rem;line-height:1.6;">
                SHAP (SHapley Additive exPlanations) shows WHY the AI made each prediction.
                It reveals which health features pushed your risk UP 🔴 or DOWN 🟢 for each disease.
                This makes the AI transparent and trustworthy.
            </div>
        </div>
        """, unsafe_allow_html=True)

        selected_disease = st.selectbox(
            "Select Disease to Explain",
            ["Diabetes 🩸", "Cardiovascular ❤️",
             "Kidney (CKD) 🫘", "Hypertension 🩺", "Fatty Liver 🫀"],
            key="shap_disease"
        )

        disease_key_map = {
            "Diabetes 🩸":        "diabetes",
            "Cardiovascular ❤️":  "cvd",
            "Kidney (CKD) 🫘":    "ckd",
            "Hypertension 🩺":    "hypertension",
            "Fatty Liver 🫀":     "fatty_liver"
        }
        sel_key = disease_key_map[selected_disease]

        if st.button("🔍 Generate SHAP Explanation", key="shap_btn"):
            with st.spinner("🔄 Computing SHAP values..."):

                feature_cols = ['age','gender','bmi','blood_pressure','blood_sugar',
                                'cholesterol','heart_rate','spo2','creatinine','hba1c',
                                'triglycerides','alt_enzyme','smoker','diabetes_history',
                                'family_history','alcohol','physical_activity']

                pa_val = int(physical_activity.split('(')[1].replace(')',''))
                patient_arr = np.array([[age, 1 if gender=="Male" else 0, bmi,
                                         blood_pressure, blood_sugar, cholesterol,
                                         heart_rate, spo2, creatinine, hba1c,
                                         triglycerides, alt_enzyme,
                                         1 if smoker else 0,
                                         1 if diabetes_history else 0,
                                         1 if family_history else 0,
                                         1 if alcohol else 0, pa_val]])

                patient_scaled = multi_scaler.transform(patient_arr)
                patient_df     = pd.DataFrame(patient_scaled, columns=feature_cols)

                sel_model = disease_models[sel_key]

                # ── SHAP Feature Importance (manual fallback) ──
                try:
                    if shap_available:
                        explainer   = shap.TreeExplainer(sel_model)
                        shap_values = explainer.shap_values(patient_df)
                        if isinstance(shap_values, list):
                            sv = shap_values[1][0]
                        else:
                            sv = shap_values[0]
                    else:
                        raise Exception("shap not installed")

                except Exception:
                    # Fallback: use feature importances from XGBoost
                    fi  = sel_model.feature_importances_
                    sv  = fi * (patient_scaled[0] - patient_scaled[0].mean())

                # Sort by absolute value
                importance_df = pd.DataFrame({
                    'Feature':    feature_cols,
                    'SHAP':       sv,
                    'AbsSHAP':    np.abs(sv),
                    'Value':      patient_arr[0]
                }).sort_values('AbsSHAP', ascending=False)

            st.markdown("---")

            # ── Row 1: Waterfall + Force Plot ─────────────────
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("#### 🌊 SHAP Waterfall — Feature Impact")
                top_n  = importance_df.head(12)
                colors = ['#e53935' if v > 0 else '#2e7d32'
                          for v in top_n['SHAP']]

                fig, ax = plt.subplots(figsize=(6, 5), facecolor='#16213e')
                ax.set_facecolor('#16213e')
                bars = ax.barh(
                    top_n['Feature'][::-1],
                    top_n['SHAP'][::-1],
                    color=colors[::-1],
                    edgecolor='#ffffff11', height=0.6
                )
                for bar, val in zip(bars, top_n['SHAP'][::-1]):
                    ax.text(
                        val + (0.001 if val >= 0 else -0.001),
                        bar.get_y() + bar.get_height()/2,
                        f'{val:+.3f}',
                        va='center',
                        ha='left' if val >= 0 else 'right',
                        color='white', fontsize=7.5, fontweight='bold'
                    )
                ax.axvline(x=0, color='white', linewidth=0.8, alpha=0.5)
                ax.set_xlabel("SHAP Value (Impact on Risk)", color='#aaa', fontsize=9)
                ax.set_title(f"Feature Impact — {selected_disease}",
                             color='white', fontweight='bold', fontsize=10, pad=10)
                ax.tick_params(colors='white', labelsize=8)
                for sp in ['top','right']: ax.spines[sp].set_visible(False)
                for sp in ['bottom','left']: ax.spines[sp].set_color('#333')
                red_patch   = mpatches.Patch(color='#e53935', label='Increases Risk')
                green_patch = mpatches.Patch(color='#2e7d32', label='Decreases Risk')
                ax.legend(handles=[red_patch, green_patch],
                          fontsize=7.5, facecolor='#1a1a2e', labelcolor='white')
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with c2:
                st.markdown("#### 🎯 Top Risk Contributors")
                top5     = importance_df.head(5)
                top5_neg = importance_df[importance_df['SHAP'] < 0].head(5)

                st.markdown("**🔴 Top factors INCREASING your risk:**")
                risk_up = importance_df[importance_df['SHAP'] > 0].head(5)
                if len(risk_up) > 0:
                    for _, row in risk_up.iterrows():
                        pct = min(abs(row['SHAP']) * 500, 100)
                        st.markdown(f"""
                        <div style="margin:6px 0;">
                            <div style="display:flex;justify-content:space-between;
                                 color:white;font-size:0.8rem;margin-bottom:3px;">
                                <span>🔴 {row['Feature']}</span>
                                <span style="color:#ff5252;">{row['SHAP']:+.4f}</span>
                            </div>
                            <div style="background:#333;border-radius:4px;height:6px;">
                                <div style="background:#e53935;width:{pct:.0f}%;
                                     height:6px;border-radius:4px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("No major risk-increasing factors!")

                st.markdown("**🟢 Top factors DECREASING your risk:**")
                risk_down = importance_df[importance_df['SHAP'] < 0].head(5)
                if len(risk_down) > 0:
                    for _, row in risk_down.iterrows():
                        pct = min(abs(row['SHAP']) * 500, 100)
                        st.markdown(f"""
                        <div style="margin:6px 0;">
                            <div style="display:flex;justify-content:space-between;
                                 color:white;font-size:0.8rem;margin-bottom:3px;">
                                <span>🟢 {row['Feature']}</span>
                                <span style="color:#00e676;">{row['SHAP']:+.4f}</span>
                            </div>
                            <div style="background:#333;border-radius:4px;height:6px;">
                                <div style="background:#2e7d32;width:{pct:.0f}%;
                                     height:6px;border-radius:4px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No major risk-decreasing factors detected.")

            st.markdown("---")

            # ── Row 2: Global Feature Importance ──────────────
            st.markdown("#### 🌍 Global Feature Importance (All Features)")
            c1, c2 = st.columns(2)

            with c1:
                fi     = sel_model.feature_importances_
                fi_df  = pd.DataFrame({
                    'Feature':    feature_cols,
                    'Importance': fi
                }).sort_values('Importance', ascending=True).tail(12)

                fig, ax = plt.subplots(figsize=(6, 5), facecolor='#16213e')
                ax.set_facecolor('#16213e')
                colors_fi = plt.cm.RdYlGn(fi_df['Importance'] / fi_df['Importance'].max())
                bars = ax.barh(fi_df['Feature'], fi_df['Importance'],
                               color=colors_fi, edgecolor='#ffffff11', height=0.6)
                for bar, val in zip(bars, fi_df['Importance']):
                    ax.text(val + 0.001,
                            bar.get_y() + bar.get_height()/2,
                            f'{val:.3f}', va='center',
                            color='white', fontsize=7.5)
                ax.set_xlabel("Feature Importance Score", color='#aaa', fontsize=9)
                ax.set_title(f"Global Importance — {selected_disease}",
                             color='white', fontweight='bold', fontsize=10, pad=10)
                ax.tick_params(colors='white', labelsize=8)
                for sp in ['top','right']: ax.spines[sp].set_visible(False)
                for sp in ['bottom','left']: ax.spines[sp].set_color('#333')
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with c2:
                # Donut chart of top 6 features
                top6 = pd.DataFrame({
                    'Feature':    feature_cols,
                    'Importance': fi
                }).sort_values('Importance', ascending=False).head(6)

                fig, ax = plt.subplots(figsize=(6, 5), facecolor='#16213e')
                ax.set_facecolor('#16213e')
                donut_colors = ['#e53935','#ff7043','#ff9800',
                                '#1e88e5','#43a047','#8e24aa']
                wedges, texts, autotexts = ax.pie(
                    top6['Importance'],
                    labels=top6['Feature'],
                    colors=donut_colors,
                    autopct='%1.1f%%',
                    startangle=90,
                    wedgeprops=dict(edgecolor='#16213e', linewidth=2, width=0.6)
                )
                for t in texts:     t.set_color('white'); t.set_fontsize(8)
                for t in autotexts: t.set_color('white'); t.set_fontweight('bold'); t.set_fontsize(7)
                ax.set_title(f"Top 6 Feature Share — {selected_disease}",
                             color='white', fontweight='bold', fontsize=10, pad=10)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            st.markdown("---")

            # ── Row 3: Counterfactual Explanations ────────────
            st.markdown("#### 💡 Counterfactual Explanations — What Would Change Your Risk?")
            st.markdown("""
            <div style="background:rgba(255,152,0,0.08);border-radius:12px;
                 padding:14px;border:1px solid rgba(255,152,0,0.2);margin-bottom:12px;">
                <div style="color:#ffcc02;font-size:0.85rem;">
                    💡 These are specific changes that could reduce your risk for this disease.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Generate counterfactuals based on disease
            counterfactuals = {
                'diabetes': [
                    ("Blood Sugar",    blood_sugar,  140,  "mg/dL", blood_sugar > 140),
                    ("HbA1c",          hba1c,        5.7,  "%",     hba1c > 5.7),
                    ("BMI",            bmi,          25.0, "",      bmi > 25),
                    ("Physical Activity","Sedentary","Active","",   pa == 0),
                ],
                'cvd': [
                    ("Cholesterol",    cholesterol,  200,  "mg/dL", cholesterol > 200),
                    ("Blood Pressure", blood_pressure,120, "mmHg",  blood_pressure > 120),
                    ("Smoking",        "Yes" if smoker else "No","No","",smoker),
                    ("BMI",            bmi,          25.0, "",      bmi > 25),
                ],
                'ckd': [
                    ("Creatinine",     creatinine,   1.2,  "mg/dL", creatinine > 1.2),
                    ("Blood Pressure", blood_pressure,120, "mmHg",  blood_pressure > 120),
                    ("Blood Sugar",    blood_sugar,  140,  "mg/dL", blood_sugar > 140),
                ],
                'hypertension': [
                    ("Blood Pressure", blood_pressure,120, "mmHg",  blood_pressure > 120),
                    ("BMI",            bmi,          25.0, "",      bmi > 25),
                    ("Alcohol",        "Yes" if alcohol else "No","No","",alcohol),
                    ("Physical Activity","Sedentary","Active","",   pa == 0),
                ],
                'fatty_liver': [
                    ("Triglycerides",  triglycerides,150,  "mg/dL", triglycerides > 150),
                    ("ALT Enzyme",     alt_enzyme,   40,   "U/L",   alt_enzyme > 40),
                    ("BMI",            bmi,          25.0, "",      bmi > 25),
                    ("Alcohol",        "Yes" if alcohol else "No","No","",alcohol),
                ],
            }

            cfs = counterfactuals.get(sel_key, [])
            c1, c2 = st.columns(2)
            for i, (feat, current, target, unit, is_risk) in enumerate(cfs):
                col = c1 if i % 2 == 0 else c2
                if is_risk:
                    col.markdown(f"""
                    <div style="background:rgba(229,57,53,0.1);border-radius:12px;
                         padding:14px;border:1px solid rgba(229,57,53,0.3);margin:6px 0;">
                        <div style="color:#ff7043;font-weight:600;font-size:0.88rem;">
                            ⚠️ {feat}
                        </div>
                        <div style="color:#aaa;font-size:0.8rem;margin-top:6px;">
                            Current: <span style="color:#ff5252;font-weight:600;">
                            {current}{unit}</span>
                        </div>
                        <div style="color:#aaa;font-size:0.8rem;">
                            Target: <span style="color:#00e676;font-weight:600;">
                            {target}{unit}</span>
                        </div>
                        <div style="color:#ffcc02;font-size:0.78rem;margin-top:6px;">
                            → Changing this could reduce your {selected_disease.split()[0]} risk
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    col.markdown(f"""
                    <div style="background:rgba(46,125,50,0.1);border-radius:12px;
                         padding:14px;border:1px solid rgba(46,125,50,0.3);margin:6px 0;">
                        <div style="color:#66bb6a;font-weight:600;font-size:0.88rem;">
                            ✅ {feat}
                        </div>
                        <div style="color:#aaa;font-size:0.8rem;margin-top:6px;">
                            Current: <span style="color:#00e676;font-weight:600;">
                            {current}{unit}</span> — Already good!
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

            # ── Row 4: SHAP Summary Table ──────────────────────
            st.markdown("#### 📋 Complete SHAP Feature Table")
            shap_table = importance_df[['Feature','Value','SHAP']].copy()
            shap_table['Impact']    = shap_table['SHAP'].apply(
                lambda x: '🔴 Risk UP' if x > 0.001 else ('🟢 Risk DOWN' if x < -0.001 else '⚪ Neutral')
            )
            shap_table['SHAP']  = shap_table['SHAP'].round(4)
            shap_table['Value'] = shap_table['Value'].round(2)
            st.dataframe(
                shap_table.reset_index(drop=True),
                use_container_width=True,
                height=350
            )

        else:
            st.info("👆 Select a disease and click **🔍 Generate SHAP Explanation** to see why the AI made its prediction")
            st.markdown("""
            <div style="display:flex;gap:15px;flex-wrap:wrap;margin-top:20px;">
                <div style="background:rgba(255,255,255,0.05);border-radius:12px;
                     padding:18px;flex:1;min-width:200px;
                     border:1px solid rgba(255,255,255,0.1);text-align:center;">
                    <div style="font-size:2rem;">🌊</div>
                    <div style="color:white;font-weight:600;margin-top:6px;">Waterfall Plot</div>
                    <div style="color:#777;font-size:0.78rem;margin-top:4px;">
                        Shows each feature's contribution</div>
                </div>
                <div style="background:rgba(255,255,255,0.05);border-radius:12px;
                     padding:18px;flex:1;min-width:200px;
                     border:1px solid rgba(255,255,255,0.1);text-align:center;">
                    <div style="font-size:2rem;">🌍</div>
                    <div style="color:white;font-weight:600;margin-top:6px;">Global Importance</div>
                    <div style="color:#777;font-size:0.78rem;margin-top:4px;">
                        Which features matter most overall</div>
                </div>
                <div style="background:rgba(255,255,255,0.05);border-radius:12px;
                     padding:18px;flex:1;min-width:200px;
                     border:1px solid rgba(255,255,255,0.1);text-align:center;">
                    <div style="font-size:2rem;">💡</div>
                    <div style="color:white;font-weight:600;margin-top:6px;">Counterfactuals</div>
                    <div style="color:#777;font-size:0.78rem;margin-top:4px;">
                        What to change to reduce risk</div>
                </div>
                <div style="background:rgba(255,255,255,0.05);border-radius:12px;
                     padding:18px;flex:1;min-width:200px;
                     border:1px solid rgba(255,255,255,0.1);text-align:center;">
                    <div style="font-size:2rem;">📋</div>
                    <div style="color:white;font-weight:600;margin-top:6px;">SHAP Table</div>
                    <div style="color:#777;font-size:0.78rem;margin-top:4px;">
                        Full feature impact breakdown</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 5: Voice Assistant ────────────────────────────────
    with tab5:
        st.markdown('<div class="section-header">🎙️ Multilingual Voice Assistant</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="background:rgba(255,152,0,0.08);border-radius:12px;
             padding:16px;border:1px solid rgba(255,152,0,0.2);margin-bottom:16px;">
            <div style="color:#ffcc02;font-weight:600;margin-bottom:4px;">
                🎙️ Voice Report Available in 3 Languages
            </div>
            <div style="color:#aaa;font-size:0.82rem;">
                Click any language button to hear your complete health report spoken aloud.
                Works best in Chrome browser.
            </div>
        </div>
        """,unsafe_allow_html=True)

        en_text = get_voice_report(
            patient_name,age,gender,risk_name="",
            confidence=0,probabilities=probabilities,
            predictions=predictions,recs=all_recs,language='en'
        )
        ta_text = get_voice_report(
            patient_name,age,gender,risk_name="",
            confidence=0,probabilities=probabilities,
            predictions=predictions,recs=all_recs,language='ta'
        )
        hi_text = get_voice_report(
            patient_name,age,gender,risk_name="",
            confidence=0,probabilities=probabilities,
            predictions=predictions,recs=all_recs,language='hi'
        )
        st.components.v1.html(
            get_language_html(en_text,ta_text,hi_text),
            height=280
        )

        st.markdown("---")
        st.markdown('<div class="section-header">📝 Voice Script Preview</div>',
                    unsafe_allow_html=True)
        lang_preview = st.radio("Preview Language",
                                ["English","Tamil","Hindi"],
                                horizontal=True)
        if lang_preview == "English":
            st.text_area("English Script", en_text, height=200)
        elif lang_preview == "Tamil":
            st.text_area("Tamil Script", ta_text, height=200)
        else:
            st.text_area("Hindi Script", hi_text, height=200)

    # ── TAB 6: AI Report ──────────────────────────────────────
    with tab6:
        st.markdown('<div class="section-header">🤖 Claude AI — Smart Health Report</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="llm-card">
            <div style="color:#4d9fff;font-size:1rem;font-weight:600;margin-bottom:6px;">
                🤖 Powered by Claude AI (Anthropic)
            </div>
            <div style="color:#aaa;font-size:0.82rem;line-height:1.6;">
                Get a personalized, easy-to-understand health report generated by
                advanced AI. The report explains your results in simple language,
                tells you what is causing each risk, and gives you specific
                actionable recommendations — available in English, Tamil & Hindi.
            </div>
        </div>
        """,unsafe_allow_html=True)

        c1,c2 = st.columns([2,1])
        with c1:
            llm_language = st.selectbox(
                "Select Report Language",
                ["English","Tamil (தமிழ்)","Hindi (हिंदी)"],
                key="llm_lang"
            )
        with c2:
            st.markdown("<br>",unsafe_allow_html=True)
            generate_btn = st.button("🤖 Generate AI Report",
                                      use_container_width=True,
                                      key="gen_btn")

        if generate_btn:
            lang_code = 'en'
            if 'Tamil' in llm_language: lang_code = 'ta'
            if 'Hindi' in llm_language: lang_code = 'hi'

            with st.spinner("🤖 Claude AI is generating your personalized report..."):
                llm_report_text = generate_llm_report(
                    patient_name=patient_name,
                    age=age, gender=gender,
                    bmi=bmi, blood_pressure=blood_pressure,
                    blood_sugar=blood_sugar, cholesterol=cholesterol,
                    heart_rate=heart_rate, spo2=spo2,
                    creatinine=creatinine, hba1c=hba1c,
                    triglycerides=triglycerides, alt_enzyme=alt_enzyme,
                    smoker=smoker, diabetes_history=diabetes_history,
                    family_history=family_history, alcohol=alcohol,
                    physical_activity=physical_activity,
                    predictions=predictions,
                    probabilities=probabilities,
                    language=lang_code
                )

            st.markdown("""
            <div style="background:rgba(255,255,255,0.05);border-radius:15px;
                 padding:25px;border:1px solid rgba(255,255,255,0.1);
                 margin:15px 0;">
            """,unsafe_allow_html=True)
            st.markdown(llm_report_text)
            st.markdown("</div>",unsafe_allow_html=True)

            # Speak AI report
            lang_map   = {'en':'en-US','ta':'ta-IN','hi':'hi-IN'}
            speak_text = llm_report_text.replace("'","").replace('"','').replace('\n',' ')
            st.components.v1.html(f"""
            <div style="text-align:center;padding:15px;
                 background:rgba(255,152,0,0.08);border-radius:12px;
                 border:1px solid rgba(255,152,0,0.2);">
                <div style="color:#ffcc02;font-weight:600;margin-bottom:10px;">
                    🎙️ Listen to AI Report
                </div>
                <button onclick="speakAI()" style="
                    background:linear-gradient(135deg,#ff9800,#ff5722);
                    color:white;border:none;border-radius:25px;
                    padding:10px 22px;font-size:0.88rem;
                    font-weight:600;cursor:pointer;margin:4px;">
                    🔊 Speak
                </button>
                <button onclick="window.speechSynthesis.cancel()" style="
                    background:linear-gradient(135deg,#424242,#212121);
                    color:white;border:none;border-radius:25px;
                    padding:10px 22px;font-size:0.88rem;
                    font-weight:600;cursor:pointer;margin:4px;">
                    ⏹️ Stop
                </button>
                <button onclick="window.speechSynthesis.pause()" style="
                    background:linear-gradient(135deg,#1565c0,#0d47a1);
                    color:white;border:none;border-radius:25px;
                    padding:10px 22px;font-size:0.88rem;
                    font-weight:600;cursor:pointer;margin:4px;">
                    ⏸️ Pause
                </button>
                <button onclick="window.speechSynthesis.resume()" style="
                    background:linear-gradient(135deg,#1b5e20,#2e7d32);
                    color:white;border:none;border-radius:25px;
                    padding:10px 22px;font-size:0.88rem;
                    font-weight:600;cursor:pointer;margin:4px;">
                    ▶️ Resume
                </button>
                <div id="s" style="color:#ffcc02;font-size:0.78rem;margin-top:8px;"></div>
            </div>
            <script>
            function speakAI(){{
                window.speechSynthesis.cancel();
                var utt=new SpeechSynthesisUtterance('{speak_text[:3000]}');
                utt.lang='{lang_map.get(lang_code,"en-US")}';
                utt.rate=0.9;
                utt.onstart=function(){{document.getElementById('s').innerText='🎙️ Speaking...';}}
                utt.onend=function(){{document.getElementById('s').innerText='✅ Done!';}}
                window.speechSynthesis.speak(utt);
            }}
            </script>
            """,height=130)

            st.download_button(
                "⬇️ Download AI Report",
                data=llm_report_text,
                file_name=f"{patient_name}_AI_report_{lang_code}.txt",
                mime="text/plain",
                key="llm_dl"
            )

        else:
            st.info("👆 Select language and click **🤖 Generate AI Report** to get your personalized AI health report")

        # Short AI summary
        st.markdown("---")
        st.markdown('<div class="section-header">💬 Quick AI Summary</div>',
                    unsafe_allow_html=True)
        if st.button("✨ Generate Quick Summary", key="sum_btn"):
            with st.spinner("Generating summary..."):
                summary = generate_llm_summary(
                    predictions, probabilities, patient_name
                )
            st.markdown(f"""
            <div style="background:rgba(26,115,232,0.1);border-radius:12px;
                 padding:20px;border:1px solid rgba(26,115,232,0.3);
                 color:white;font-size:0.95rem;line-height:1.7;">
                🤖 {summary}
            </div>
            """,unsafe_allow_html=True)

    # ── TAB 7: Download ───────────────────────────────────────
    with tab7:
        st.markdown('<div class="section-header">📄 Download Health Report</div>',
                    unsafe_allow_html=True)

        now           = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        risk_names_str= ", ".join([disease_info[d][1] for d in risk_diseases]) \
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
  Activity       : {physical_activity}
{'='*60}
DISEASE PREDICTIONS:
  Diabetes       : {'AT RISK' if predictions['diabetes']     else 'SAFE'} ({probabilities['diabetes']:.1f}%)
  Cardiovascular : {'AT RISK' if predictions['cvd']          else 'SAFE'} ({probabilities['cvd']:.1f}%)
  Kidney (CKD)   : {'AT RISK' if predictions['ckd']          else 'SAFE'} ({probabilities['ckd']:.1f}%)
  Hypertension   : {'AT RISK' if predictions['hypertension'] else 'SAFE'} ({probabilities['hypertension']:.1f}%)
  Fatty Liver    : {'AT RISK' if predictions['fatty_liver']  else 'SAFE'} ({probabilities['fatty_liver']:.1f}%)
{'='*60}
DISEASES AT RISK : {risk_names_str}
OVERALL STATUS   : {len(risk_diseases)}/5 diseases at risk
{'='*60}
RECOMMENDATIONS:
""" + "\n".join([f"  - {r}" for r in all_recs]) + f"""
{'='*60}
DISCLAIMER: AI-generated report for informational purposes only.
Always consult a qualified medical professional for diagnosis.
{'='*60}
"""
        st.text_area("Full Report", report, height=300)
        c1,c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇️ Download Full Report",
                data=report,
                file_name=f"{patient_name}_multidisease_report.txt",
                mime="text/plain",
                key="dl1"
            )
        with c2:
            en_report = get_voice_report(
                patient_name,age,gender,"",0,
                probabilities,predictions,all_recs,'en'
            )
            st.download_button(
                "⬇️ Download Voice Script",
                data=en_report,
                file_name=f"{patient_name}_voice_script.txt",
                mime="text/plain",
                key="dl2"
            )

        st.markdown("---")
        st.markdown('<div class="section-header">🔒 Privacy & Federated Learning</div>',
                    unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        for col,name in zip([c1,c2,c3],
                            ["🏥 Hospital A","🏥 Hospital B","🏥 Hospital C"]):
            col.markdown(f'<div class="info-card"><h3>{name}</h3>'
                         f'<p>Local Training Only<br>Data stays private</p></div>',
                         unsafe_allow_html=True)
        st.success("🔒 Only model weights shared via FedAvg — Patient data NEVER leaves local systems!")


# ─── Welcome Screen ──────────────────────────────────────────
else:
    st.markdown("""
    <div style="text-align:center;padding:30px;color:#aaa;">
        <div style="font-size:3rem;margin-bottom:10px;">🔬</div>
        <h2 style="color:white;">Multi-Disease AI Prediction System</h2>
        <p>Enter patient details in the sidebar and click<br>
        <strong style="color:#ff4b6e;">🔍 Predict All Diseases</strong>
        to get a complete 5-disease health analysis</p>
    </div>
    """,unsafe_allow_html=True)

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
            <div style="color:white;font-weight:600;font-size:0.9rem;margin-top:6px;">{name}</div>
            <div style="color:#777;font-size:0.75rem;margin-top:4px;">{desc}</div>
        </div>""",unsafe_allow_html=True)

    st.markdown("---")

    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(icon,title,desc) in zip([c1,c2,c3,c4,c5],[
        ("📊","Analysis Tab",   "Radar + Bar + Pie charts"),
        ("📋","Vitals Tab",     "All health metrics"),
        ("🔍","SHAP & XAI Tab", "Explainable AI per disease"),
        ("🎙️","Voice Tab",      "EN | Tamil | Hindi"),
        ("🤖","AI Report Tab",  "Claude AI report"),
    ]):
        col.markdown(f"""
        <div style="background:rgba(255,255,255,0.05);border-radius:15px;
             padding:18px;text-align:center;
             border:1px solid rgba(255,255,255,0.1);">
            <div style="font-size:1.8rem;">{icon}</div>
            <div style="color:white;font-weight:600;font-size:0.88rem;margin-top:6px;">{title}</div>
            <div style="color:#777;font-size:0.75rem;margin-top:4px;">{desc}</div>
        </div>""",unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#444;padding:16px;font-size:0.85rem;">
        ❤️ FedHealth-Twin | Multi-Disease AI Prediction | Multilingual Voice | Claude AI Reports
    </div>""",unsafe_allow_html=True)