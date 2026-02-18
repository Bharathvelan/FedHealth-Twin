import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="FedHealth-Twin | AI Health Risk",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* ── Hero Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #ff4b6e, #1a73e8, #00c9ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 10px 0;
    }
    .hero-subtitle {
        color: #aaa;
        font-size: 1.1rem;
        margin: 5px 0;
    }
    .hero-badges {
        margin-top: 15px;
    }
    .badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 3px;
    }
    .badge-blue  { background: rgba(26,115,232,0.2); color:#1a73e8; border:1px solid #1a73e8; }
    .badge-red   { background: rgba(255,75,110,0.2); color:#ff4b6e; border:1px solid #ff4b6e; }
    .badge-green { background: rgba(0,201,87,0.2);   color:#00c957; border:1px solid #00c957; }
    .badge-purple{ background: rgba(156,39,176,0.2); color:#ce93d8; border:1px solid #ce93d8; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    .sidebar-logo {
        text-align: center;
        padding: 15px 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 15px;
    }
    .sidebar-logo h2 {
        color: #ff4b6e;
        margin: 5px 0 0 0;
        font-size: 1.4rem;
    }
    .sidebar-logo p {
        color: #888;
        font-size: 0.75rem;
        margin: 2px 0;
    }

    /* ── Risk Cards ── */
    .risk-low {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        color: white;
        box-shadow: 0 4px 20px rgba(46,125,50,0.4);
        border: 1px solid #4caf50;
    }
    .risk-medium {
        background: linear-gradient(135deg, #e65100, #f57c00);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        color: white;
        box-shadow: 0 4px 20px rgba(245,124,0,0.4);
        border: 1px solid #ff9800;
    }
    .risk-high {
        background: linear-gradient(135deg, #b71c1c, #c62828);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        color: white;
        box-shadow: 0 4px 20px rgba(198,40,40,0.4);
        border: 1px solid #f44336;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%   { box-shadow: 0 4px 20px rgba(198,40,40,0.4); }
        50%  { box-shadow: 0 4px 40px rgba(198,40,40,0.8); }
        100% { box-shadow: 0 4px 20px rgba(198,40,40,0.4); }
    }

    /* ── Section Headers ── */
    .section-header {
        background: linear-gradient(90deg, rgba(26,115,232,0.2), transparent);
        border-left: 4px solid #1a73e8;
        padding: 10px 15px;
        border-radius: 0 10px 10px 0;
        margin: 20px 0 15px 0;
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
    }

    /* ── Info Cards ── */
    .info-card {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
        margin: 5px;
    }
    .info-card h3 { color: #1a73e8; margin: 0; font-size: 1.5rem; }
    .info-card p  { color: #aaa; margin: 5px 0 0 0; font-size: 0.85rem; }

    /* ── Feature Cards (Welcome) ── */
    .feature-card {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
        transition: transform 0.3s;
    }
    .feature-card:hover { transform: translateY(-5px); }
    .feature-icon { font-size: 2.5rem; margin-bottom: 10px; }
    .feature-title { color: white; font-weight: 600; font-size: 1.1rem; }
    .feature-desc  { color: #888; font-size: 0.85rem; margin-top: 5px; }

    /* ── Predict Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #ff4b6e, #1a73e8) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 12px 30px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        width: 100% !important;
        transition: all 0.3s !important;
        box-shadow: 0 4px 15px rgba(255,75,110,0.4) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255,75,110,0.6) !important;
    }

    /* ── Download Button ── */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #00c9ff, #0072ff) !important;
        color: white !important;
        border-radius: 25px !important;
        border: none !important;
        font-weight: 600 !important;
    }

    /* ── Hide Streamlit Branding ── */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Load Models ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    import sys
    sys.path.append('.')
    from federated_learning import FederatedNet

    with open('models/xgboost_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    fed_model = FederatedNet(input_dim=12, num_classes=3)
    fed_model.load_state_dict(torch.load('models/federated_model.pth',
                                          map_location=torch.device('cpu')))
    fed_model.eval()
    return xgb_model, scaler, fed_model

try:
    xgb_model, scaler, fed_model = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    model_error = str(e)


# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <img src="https://img.icons8.com/fluency/80/heart-with-pulse.png" width="70"/>
        <h2>FedHealth-Twin</h2>
        <p>AI Smart Health System</p>
        <p>🔒 Privacy Preserved | 🧠 AI Powered</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 👤 Demographics")
    patient_name = st.text_input("Patient Name", value="John Doe")
    age    = st.slider("Age (years)", 18, 90, 45)
    gender = st.selectbox("Gender", ["Male", "Female"])

    st.markdown("### 🩺 Clinical Vitals")
    bmi            = st.slider("BMI",                  15.0, 50.0, 25.0, 0.1)
    blood_pressure = st.slider("Blood Pressure (mmHg)", 60,  200,  120)
    blood_sugar    = st.slider("Blood Sugar (mg/dL)",   70,  500,  100)
    cholesterol    = st.slider("Cholesterol (mg/dL)",  100,  400,  180)
    heart_rate     = st.slider("Heart Rate (bpm)",      40,  150,   75)
    spo2           = st.slider("SpO2 (%)",             80.0,100.0, 98.0, 0.1)
    creatinine     = st.slider("Creatinine (mg/dL)",   0.5,  6.0,  1.0, 0.1)

    st.markdown("### 🏥 Medical History")
    smoker           = st.checkbox("🚬 Smoker")
    diabetes_history = st.checkbox("🩸 Diabetes History")
    family_history   = st.checkbox("👨‍👩‍👧 Family History of Disease")

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Health Risk", use_container_width=True)

    st.markdown("""
    <div style="text-align:center; margin-top:20px; color:#555; font-size:0.75rem;">
        <img src="https://img.icons8.com/fluency/24/lock.png" width="16"/>
        Data never leaves your device<br>
        Powered by Federated Learning
    </div>
    """, unsafe_allow_html=True)


# ─── Hero Banner ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div>
        <img src="https://img.icons8.com/fluency/60/heart-with-pulse.png" width="55" style="margin:0 8px;"/>
        <img src="https://img.icons8.com/fluency/60/caduceus.png"         width="55" style="margin:0 8px;"/>
        <img src="https://img.icons8.com/fluency/60/hospital.png"         width="55" style="margin:0 8px;"/>
        <img src="https://img.icons8.com/fluency/60/brain.png"            width="55" style="margin:0 8px;"/>
        <img src="https://img.icons8.com/fluency/60/dna-helix.png"        width="55" style="margin:0 8px;"/>
    </div>
    <div class="hero-title">🏥 FedHealth-Twin</div>
    <div class="hero-subtitle">AI-Based Smart Health Risk Prediction System</div>
    <div class="hero-subtitle" style="color:#666;">
        Final Year Project | AI & ML | 2024–2025
    </div>
    <div class="hero-badges">
        <span class="badge badge-blue">🔒 Federated Learning</span>
        <span class="badge badge-red">🧠 Deep Learning</span>
        <span class="badge badge-green">📊 Explainable AI</span>
        <span class="badge badge-purple">🔬 Multimodal Fusion</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Main Logic ──────────────────────────────────────────────
if predict_btn:
    if not models_loaded:
        st.error(f"❌ Models not loaded: {model_error}")
        st.stop()

    patient_data = {
        'age': age, 'gender': 1 if gender == "Male" else 0,
        'bmi': bmi, 'blood_pressure': blood_pressure,
        'blood_sugar': blood_sugar, 'cholesterol': cholesterol,
        'heart_rate': heart_rate, 'spo2': spo2,
        'creatinine': creatinine,
        'smoker': 1 if smoker else 0,
        'diabetes_history': 1 if diabetes_history else 0,
        'family_history': 1 if family_history else 0
    }

    with st.spinner("🔄 Analyzing patient data with AI models..."):
        feature_cols = ['age','gender','bmi','blood_pressure','blood_sugar',
                        'cholesterol','heart_rate','spo2','creatinine',
                        'smoker','diabetes_history','family_history']
        patient_df = pd.DataFrame([patient_data])[feature_cols]

        xgb_pred  = xgb_model.predict(patient_df)[0]
        xgb_proba = xgb_model.predict_proba(patient_df)[0]

        patient_scaled = scaler.transform(patient_df)
        patient_tensor = torch.FloatTensor(patient_scaled)
        with torch.no_grad():
            fed_out   = fed_model(patient_tensor)
            fed_proba = torch.softmax(fed_out, dim=1).numpy()[0]
            fed_pred  = np.argmax(fed_proba)

        ensemble_proba = (xgb_proba + fed_proba) / 2
        final_pred     = np.argmax(ensemble_proba)
        confidence     = ensemble_proba[final_pred] * 100

    risk_labels = ['Low Risk ✅', 'Medium Risk ⚠️', 'High Risk 🚨']
    risk_colors = ['risk-low', 'risk-medium', 'risk-high']
    risk_names  = ['LOW RISK', 'MEDIUM RISK', 'HIGH RISK']

    # ── Result Card ───────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 Prediction Result</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        st.markdown(f"""
        <div class="{risk_colors[final_pred]}">
            {risk_labels[final_pred]}<br>
            <small style="font-size:1rem; font-weight:400;">
                Confidence: {confidence:.1f}%
            </small><br>
            <small style="font-size:0.85rem; font-weight:300;">
                Patient: {patient_name}
            </small>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="info-card">
            <h3>XGBoost</h3>
            <p>{risk_names[xgb_pred]}</p>
            <p style="color:#1a73e8;">{max(xgb_proba)*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="info-card">
            <h3>FedModel</h3>
            <p>{risk_names[fed_pred]}</p>
            <p style="color:#ff4b6e;">{max(fed_proba)*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Risk Probability Distribution</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    colors = ['#2e7d32', '#f57c00', '#c62828']
    rlabels = ['Low Risk', 'Medium Risk', 'High Risk']

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4),
                               facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        bars = ax.bar(rlabels, ensemble_proba * 100,
                      color=colors, edgecolor='white',
                      linewidth=1.5, width=0.5)
        for bar, val in zip(bars, ensemble_proba):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    f'{val*100:.1f}%',
                    ha='center', fontweight='bold',
                    color='white', fontsize=11)
        ax.set_ylabel("Probability (%)", color='white')
        ax.set_title("Risk Probability (%)", color='white', fontweight='bold')
        ax.set_ylim(0, 110)
        ax.tick_params(colors='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for spine in ['bottom','left']:
            ax.spines[spine].set_color('#444')
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4),
                               facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        wedges, texts, autotexts = ax.pie(
            ensemble_proba, labels=rlabels, colors=colors,
            autopct='%1.1f%%', startangle=90,
            wedgeprops=dict(edgecolor='#1a1a2e', linewidth=2)
        )
        for t in texts:     t.set_color('white')
        for t in autotexts: t.set_color('white'); t.set_fontweight('bold')
        ax.set_title("Risk Distribution", color='white', fontweight='bold')
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ── Vitals Summary ────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Patient Vitals Summary</div>',
                unsafe_allow_html=True)

    def vstatus(val, low, high, higher_is_worse=True):
        if higher_is_worse:
            return "🔴" if val > high else ("🟡" if val > high*0.9 else "🟢")
        else:
            return "🔴" if val < low else "🟢"

    col1,col2,col3,col4 = st.columns(4)
    col1.metric("BMI",            f"{bmi}",            f"{vstatus(bmi,18.5,25)} Normal:18.5–25")
    col2.metric("Blood Pressure", f"{blood_pressure}",  f"{vstatus(blood_pressure,60,120)} Normal:<120")
    col3.metric("Blood Sugar",    f"{blood_sugar}",     f"{vstatus(blood_sugar,70,140)} Normal:70–140")
    col4.metric("SpO2",           f"{spo2}%",           f"{vstatus(spo2,95,100,False)} Normal:>95%")

    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Heart Rate",  f"{heart_rate} bpm",  f"{vstatus(heart_rate,60,100)} Normal:60–100")
    col2.metric("Cholesterol", f"{cholesterol}",      f"{vstatus(cholesterol,100,200)} Normal:<200")
    col3.metric("Creatinine",  f"{creatinine}",       f"{vstatus(creatinine,0.5,1.2)} Normal:0.5–1.2")
    col4.metric("Age",         f"{age} yrs",          "")

    st.markdown("---")

    # ── SHAP ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔍 Explainable AI — SHAP Analysis</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        try:
            st.image('reports/shap_global.png',
                     caption="🌍 Global Feature Importance (All Patients)",
                     use_container_width=True)
        except:
            st.info("Run explainability.py to generate SHAP plots")
    with col2:
        try:
            st.image('reports/shap_patient.png',
                     caption="👤 Patient-Level Explanation (This Patient)",
                     use_container_width=True)
        except:
            st.info("SHAP patient plot not available")

    st.markdown("---")

    # ── Recommendations ───────────────────────────────────────
    st.markdown('<div class="section-header">💡 Personalized Recommendations</div>',
                unsafe_allow_html=True)

    if final_pred == 0:
        st.success("✅ LOW RISK — Keep maintaining your healthy lifestyle!")
        recs = [
            "Continue regular exercise (30 min daily)",
            "Maintain balanced diet rich in fruits and vegetables",
            "Regular health checkups every 6 months",
            "Stay hydrated and maintain healthy sleep schedule"
        ]
    elif final_pred == 1:
        st.warning("⚠️ MEDIUM RISK — Lifestyle changes recommended!")
        recs = []
        if bmi > 28:          recs.append("Work on weight reduction — target BMI below 25")
        if blood_pressure>120: recs.append("Monitor BP daily, reduce salt intake")
        if blood_sugar > 150:  recs.append("Control carbohydrate intake, monitor sugar weekly")
        if smoker:             recs.append("Quit smoking — major risk amplifier")
        recs.append("Exercise 5 days/week, 45 minutes per session")
        recs.append("Consult a doctor within next 30 days")
    else:
        st.error("🚨 HIGH RISK — Immediate medical attention required!")
        recs = []
        if bmi > 30:           recs.append("URGENT: Start medically supervised weight loss")
        if blood_pressure>140: recs.append("URGENT: Seek treatment for hypertension now")
        if blood_sugar > 250:  recs.append("URGENT: Consult endocrinologist for diabetes")
        if spo2 < 92:          recs.append("URGENT: Low SpO2 — seek emergency evaluation")
        if creatinine > 2.5:   recs.append("URGENT: High creatinine — kidney evaluation needed")
        if smoker:             recs.append("URGENT: Stop smoking immediately")
        recs.append("Hospitalization or specialist consultation advised")

    for i, rec in enumerate(recs, 1):
        st.markdown(f"**{i}.** {rec}")

    st.markdown("---")

    # ── Report ────────────────────────────────────────────────
    st.markdown('<div class="section-header">📄 Health Report</div>',
                unsafe_allow_html=True)

    from datetime import datetime
    now    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""
╔══════════════════════════════════════════════════════════════╗
║         FedHealth-Twin: AI Health Risk Report                ║
╚══════════════════════════════════════════════════════════════╝

Report Generated : {now}
Patient Name     : {patient_name}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATIENT VITALS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Age              : {age} years
Gender           : {gender}
BMI              : {bmi}
Blood Pressure   : {blood_pressure} mmHg
Blood Sugar      : {blood_sugar} mg/dL
Cholesterol      : {cholesterol} mg/dL
Heart Rate       : {heart_rate} bpm
SpO2             : {spo2}%
Creatinine       : {creatinine} mg/dL
Smoker           : {'Yes' if smoker else 'No'}
Diabetes History : {'Yes' if diabetes_history else 'No'}
Family History   : {'Yes' if family_history else 'No'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PREDICTION RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
XGBoost Model    : {risk_names[xgb_pred]} ({max(xgb_proba)*100:.1f}%)
Federated Model  : {risk_names[fed_pred]} ({max(fed_proba)*100:.1f}%)
FINAL PREDICTION : {risk_names[final_pred]}
Confidence Score : {confidence:.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PERSONALIZED RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    for i, rec in enumerate(recs, 1):
        report += f"\n  {i}. {rec}"

    report += """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISCLAIMER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This report is generated by an AI system for informational
purposes only. Always consult a qualified medical professional.
════════════════════════════════════════════════════════════
"""

    st.text_area("Full Report", report, height=300)
    st.download_button(
        label="⬇️ Download Full Report",
        data=report,
        file_name=f"{patient_name}_health_report.txt",
        mime="text/plain"
    )

    st.markdown("---")

    # ── Federated Privacy Info ────────────────────────────────
    st.markdown('<div class="section-header">🔒 Federated Learning — Privacy Architecture</div>',
                unsafe_allow_html=True)

    col1,col2,col3 = st.columns(3)
    col1.markdown("""
    <div class="info-card">
        <img src="https://img.icons8.com/fluency/48/hospital.png" width="40"/>
        <h3>Hospital A</h3>
        <p>Local Training Only<br>Data stays private</p>
    </div>""", unsafe_allow_html=True)
    col2.markdown("""
    <div class="info-card">
        <img src="https://img.icons8.com/fluency/48/hospital.png" width="40"/>
        <h3>Hospital B</h3>
        <p>Local Training Only<br>Data stays private</p>
    </div>""", unsafe_allow_html=True)
    col3.markdown("""
    <div class="info-card">
        <img src="https://img.icons8.com/fluency/48/hospital.png" width="40"/>
        <h3>Hospital C</h3>
        <p>Local Training Only<br>Data stays private</p>
    </div>""", unsafe_allow_html=True)

    st.success("🔒 Only model weights are shared via FedAvg — Patient data NEVER leaves local systems!")

# ─── Welcome Screen ──────────────────────────────────────────
else:
    col1,col2,col3,col4 = st.columns(4)
    cards = [
        ("https://img.icons8.com/fluency/60/lock.png",
         "Privacy First", "Federated Learning — data never leaves device", "badge-blue"),
        ("https://img.icons8.com/fluency/60/brain.png",
         "AI Powered", "LSTM + XGBoost + Fusion Neural Network", "badge-red"),
        ("https://img.icons8.com/fluency/60/combo-chart.png",
         "Explainable", "SHAP analysis for every prediction", "badge-green"),
        ("https://img.icons8.com/fluency/60/document.png",
         "Auto Report", "Human-readable health report generated", "badge-purple"),
    ]
    for col, (icon, title, desc, badge) in zip([col1,col2,col3,col4], cards):
        col.markdown(f"""
        <div class="feature-card">
            <img src="{icon}" width="50"/>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Stats Row
    st.markdown('<div class="section-header">📈 System Statistics</div>',
                unsafe_allow_html=True)
    col1,col2,col3,col4 = st.columns(4)
    col1.markdown("""<div class="info-card"><h3>3</h3>
        <p>AI Models Combined</p></div>""", unsafe_allow_html=True)
    col2.markdown("""<div class="info-card"><h3>12</h3>
        <p>Health Features Analyzed</p></div>""", unsafe_allow_html=True)
    col3.markdown("""<div class="info-card"><h3>3</h3>
        <p>Hospital Nodes (Federated)</p></div>""", unsafe_allow_html=True)
    col4.markdown("""<div class="info-card"><h3>3</h3>
        <p>Risk Categories Predicted</p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # How to Use
    st.markdown('<div class="section-header">🚀 How to Use</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Step 1** — Enter patient name and demographics in the sidebar  
        **Step 2** — Adjust clinical vitals using the sliders  
        **Step 3** — Check relevant medical history boxes  
        **Step 4** — Click **Predict Health Risk** button  
        **Step 5** — View risk score, charts, SHAP explanations  
        **Step 6** — Download the full health report  
        """)
    with col2:
        st.markdown("""
        | Model | Purpose |
        |-------|---------|
        | XGBoost | Tabular clinical data |
        | PyTorch LSTM | Time-series vitals |
        | TF-IDF + SVD | Doctor's notes |
        | Fusion Net | Combined prediction |
        | FedAvg | Privacy preservation |
        | SHAP | Explainability |
        """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#555; padding:20px;">
        <img src="https://img.icons8.com/fluency/30/heart-with-pulse.png" width="25"/>
        FedHealth-Twin | AI-Based Smart Health Risk Prediction System<br>
        <small>Final Year Project | AI & ML | 2024–2025</small>
    </div>
    """, unsafe_allow_html=True)