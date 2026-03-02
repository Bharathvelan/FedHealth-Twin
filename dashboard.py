import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
import warnings
import json
import base64
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
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; }

.hero-banner {
    background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
    border-radius: 20px; padding: 30px 30px 22px 30px;
    text-align: center; margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.hero-icon-row {
    display: flex; justify-content: center;
    align-items: center; gap: 22px; margin-bottom: 12px;
}
.hero-icon {
    width: 58px; height: 58px; border-radius: 50%;
    display: inline-flex; align-items: center;
    justify-content: center; font-size: 1.7rem;
}
.icon-red    { background:rgba(255,75,110,0.18);  border:1.5px solid #ff4b6e; }
.icon-blue   { background:rgba(26,115,232,0.18);  border:1.5px solid #1a73e8; }
.icon-green  { background:rgba(0,201,87,0.18);    border:1.5px solid #00c957; }
.icon-purple { background:rgba(156,39,176,0.18);  border:1.5px solid #9c27b0; }
.icon-cyan   { background:rgba(0,188,212,0.18);   border:1.5px solid #00bcd4; }

.hero-title {
    font-size: 2.9rem; font-weight: 700;
    background: linear-gradient(90deg, #ff4b6e, #1a73e8, #00c9ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 8px 0 4px 0;
}
.hero-subtitle { color:#999; font-size:1.0rem; margin:4px 0; }
.hero-badges   { margin-top:12px; }
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
    background:linear-gradient(180deg,#1a1a2e 0%,#16213e 100%);
    border-right:1px solid rgba(255,255,255,0.08);
}
.sidebar-logo {
    text-align:center; padding:16px 0 14px 0;
    border-bottom:1px solid rgba(255,255,255,0.08); margin-bottom:14px;
}
.sidebar-logo h2 { color:#ff4b6e; margin:4px 0 2px 0; font-size:1.3rem; }
.sidebar-logo p  { color:#777; font-size:0.72rem; margin:2px 0; }

.risk-low {
    background:linear-gradient(135deg,#1b5e20,#2e7d32); border-radius:15px;
    padding:25px; text-align:center; font-size:1.35rem; font-weight:700; color:white;
    box-shadow:0 4px 20px rgba(46,125,50,0.45); border:1px solid #4caf50;
}
.risk-medium {
    background:linear-gradient(135deg,#bf360c,#e64a19); border-radius:15px;
    padding:25px; text-align:center; font-size:1.35rem; font-weight:700; color:white;
    box-shadow:0 4px 20px rgba(230,74,25,0.45); border:1px solid #ff7043;
}
.risk-high {
    background:linear-gradient(135deg,#b71c1c,#c62828); border-radius:15px;
    padding:25px; text-align:center; font-size:1.35rem; font-weight:700; color:white;
    border:1px solid #f44336; animation:pulse 1.5s infinite;
}
@keyframes pulse {
    0%  { box-shadow:0 4px 20px rgba(198,40,40,0.4); }
    50% { box-shadow:0 4px 45px rgba(198,40,40,0.85); }
    100%{ box-shadow:0 4px 20px rgba(198,40,40,0.4); }
}

.section-header {
    background:linear-gradient(90deg,rgba(26,115,232,0.2),transparent);
    border-left:4px solid #1a73e8; padding:10px 16px;
    border-radius:0 10px 10px 0; margin:22px 0 14px 0;
    color:white; font-size:1.1rem; font-weight:600;
}
.info-card {
    background:rgba(255,255,255,0.05); border-radius:12px; padding:18px;
    border:1px solid rgba(255,255,255,0.1); text-align:center; margin:4px;
}
.info-card h3 { color:#1a73e8; margin:0; font-size:1.5rem; }
.info-card p  { color:#aaa; margin:5px 0 0 0; font-size:0.82rem; }

.vital-card {
    background:rgba(255,255,255,0.05); border-radius:12px; padding:16px 12px;
    border:1px solid rgba(255,255,255,0.1); text-align:center; margin:4px;
}
.vital-label  { color:#aaa; font-size:0.78rem; margin-bottom:4px; }
.vital-value  { font-size:1.55rem; font-weight:700; color:white; margin:2px 0; }
.vital-normal { font-size:0.72rem; margin-top:5px; font-weight:600; }
.vital-ok     { color:#00e676; }
.vital-warn   { color:#ffab40; }
.vital-danger { color:#ff5252; }

/* ── Voice Assistant Card ── */
.voice-card {
    background: linear-gradient(135deg, #0d1b2a, #1b2838);
    border-radius: 18px; padding: 25px;
    border: 1px solid rgba(255,152,0,0.3);
    box-shadow: 0 4px 25px rgba(255,152,0,0.15);
    text-align: center; margin: 10px 0;
}
.voice-title {
    color: #ffcc02; font-size: 1.2rem;
    font-weight: 700; margin-bottom: 8px;
}
.voice-subtitle { color:#aaa; font-size:0.85rem; margin-bottom:16px; }

/* Speaking animation */
.speaking-wave {
    display: flex; justify-content: center;
    align-items: center; gap: 4px; margin: 12px 0;
}
.wave-bar {
    width: 5px; border-radius: 3px;
    background: linear-gradient(#ffcc02, #ff9800);
    animation: wave 1.2s ease-in-out infinite;
}
.wave-bar:nth-child(1){ height:12px; animation-delay:0.0s; }
.wave-bar:nth-child(2){ height:24px; animation-delay:0.1s; }
.wave-bar:nth-child(3){ height:18px; animation-delay:0.2s; }
.wave-bar:nth-child(4){ height:30px; animation-delay:0.3s; }
.wave-bar:nth-child(5){ height:22px; animation-delay:0.2s; }
.wave-bar:nth-child(6){ height:16px; animation-delay:0.1s; }
.wave-bar:nth-child(7){ height:10px; animation-delay:0.0s; }
@keyframes wave {
    0%,100%{ transform:scaleY(0.5); opacity:0.6; }
    50%    { transform:scaleY(1.5); opacity:1.0; }
}

.feature-card {
    background:rgba(255,255,255,0.05); border-radius:15px; padding:25px;
    border:1px solid rgba(255,255,255,0.1); text-align:center;
}
.feature-title { color:white; font-weight:600; font-size:1.0rem; margin-top:8px; }
.feature-desc  { color:#777; font-size:0.8rem; margin-top:5px; }

.stButton > button {
    background:linear-gradient(135deg,#ff4b6e,#1a73e8) !important;
    color:white !important; border:none !important; border-radius:25px !important;
    padding:12px 30px !important; font-weight:600 !important;
    font-size:1rem !important; width:100% !important;
    box-shadow:0 4px 15px rgba(255,75,110,0.4) !important;
}
.stDownloadButton > button {
    background:linear-gradient(135deg,#00c9ff,#0072ff) !important;
    color:white !important; border-radius:25px !important;
    border:none !important; font-weight:600 !important;
}
#MainMenu { visibility:hidden; }
footer     { visibility:hidden; }
header     { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Voice Assistant JS (Web Speech API — no extra library needed) ─
def voice_assistant_js(text, autoplay=False):
    """Inject JavaScript to speak the given text using browser TTS."""
    # Escape special characters
    text = text.replace("'", "\\'").replace('"', '\\"').replace('\n', ' ')
    auto = "true" if autoplay else "false"
    return f"""
    <div class="voice-card">
       
        <div class="speaking-wave" id="wave" style="display:none;">
            <div class="wave-bar"></div><div class="wave-bar"></div>
            <div class="wave-bar"></div><div class="wave-bar"></div>
            <div class="wave-bar"></div><div class="wave-bar"></div>
            <div class="wave-bar"></div>
        </div>
        <div style="display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-top:10px;">
            <button onclick="speakText()" style="
                background:linear-gradient(135deg,#ff9800,#ff5722);
                color:white; border:none; border-radius:25px;
                padding:10px 24px; font-size:0.95rem; font-weight:600;
                cursor:pointer; box-shadow:0 4px 15px rgba(255,152,0,0.4);">
                🔊 Speak Health Report
            </button>
            <button onclick="stopText()" style="
                background:linear-gradient(135deg,#424242,#212121);
                color:white; border:none; border-radius:25px;
                padding:10px 24px; font-size:0.95rem; font-weight:600;
                cursor:pointer;">
                ⏹️ Stop
            </button>
            <button onclick="pauseText()" style="
                background:linear-gradient(135deg,#1565c0,#0d47a1);
                color:white; border:none; border-radius:25px;
                padding:10px 24px; font-size:0.95rem; font-weight:600;
                cursor:pointer;">
                ⏸️ Pause
            </button>
            <button onclick="resumeText()" style="
                background:linear-gradient(135deg,#2e7d32,#1b5e20);
                color:white; border:none; border-radius:25px;
                padding:10px 24px; font-size:0.95rem; font-weight:600;
                cursor:pointer;">
                ▶️ Resume
            </button>
        </div>
        <div style="margin-top:12px;display:flex;gap:10px;justify-content:center;align-items:center;flex-wrap:wrap;">
            <label style="color:#aaa;font-size:0.8rem;">Speed:</label>
            <input type="range" id="rateSlider" min="0.5" max="2" step="0.1" value="0.9"
                   style="width:100px;" oninput="updateRate(this.value)"/>
            <span id="rateVal" style="color:#ffcc02;font-size:0.8rem;">0.9x</span>
            <label style="color:#aaa;font-size:0.8rem;margin-left:10px;">Pitch:</label>
            <input type="range" id="pitchSlider" min="0.5" max="2" step="0.1" value="1.0"
                   style="width:100px;" oninput="updatePitch(this.value)"/>
            <span id="pitchVal" style="color:#ffcc02;font-size:0.8rem;">1.0</span>
        </div>
        <div style="margin-top:10px;">
            <label style="color:#aaa;font-size:0.8rem;">Voice:</label>
            <select id="voiceSelect" style="
                background:#1a1a2e; color:white; border:1px solid #333;
                border-radius:8px; padding:4px 8px; margin-left:8px; font-size:0.8rem;">
            </select>
        </div>
        <div id="statusMsg" style="color:#ffcc02;font-size:0.8rem;margin-top:8px;"></div>
    </div>

    <script>
    var speechText = '{text}';
    var currentRate  = 0.9;
    var currentPitch = 1.0;
    var utterance    = null;

    function updateRate(val) {{
        currentRate = parseFloat(val);
        document.getElementById('rateVal').innerText = val + 'x';
    }}
    function updatePitch(val) {{
        currentPitch = parseFloat(val);
        document.getElementById('pitchVal').innerText = val;
    }}

    // Populate voice list
    function loadVoices() {{
        var select = document.getElementById('voiceSelect');
        if (!select) return;
        select.innerHTML = '';
        var voices = window.speechSynthesis.getVoices();
        voices.forEach(function(v, i) {{
            var opt = document.createElement('option');
            opt.value = i;
            opt.text  = v.name + ' (' + v.lang + ')';
            if (v.lang.startsWith('en')) select.appendChild(opt);
        }});
    }}
    window.speechSynthesis.onvoiceschanged = loadVoices;
    loadVoices();

    function speakText() {{
        window.speechSynthesis.cancel();
        utterance = new SpeechSynthesisUtterance(speechText);
        utterance.rate  = currentRate;
        utterance.pitch = currentPitch;
        var select = document.getElementById('voiceSelect');
        var voices = window.speechSynthesis.getVoices();
        if (select && voices.length > 0) {{
            utterance.voice = voices[parseInt(select.value) || 0];
        }}
        utterance.onstart = function() {{
            document.getElementById('wave').style.display = 'flex';
            document.getElementById('statusMsg').innerText = '🎙️ Speaking...';
        }};
        utterance.onend = function() {{
            document.getElementById('wave').style.display = 'none';
            document.getElementById('statusMsg').innerText = '✅ Done speaking!';
        }};
        utterance.onerror = function(e) {{
            document.getElementById('wave').style.display = 'none';
            document.getElementById('statusMsg').innerText = '❌ Error: ' + e.error;
        }};
        window.speechSynthesis.speak(utterance);
    }}
    function stopText() {{
        window.speechSynthesis.cancel();
        document.getElementById('wave').style.display = 'none';
        document.getElementById('statusMsg').innerText = '⏹️ Stopped.';
    }}
    function pauseText() {{
        window.speechSynthesis.pause();
        document.getElementById('wave').style.display = 'none';
        document.getElementById('statusMsg').innerText = '⏸️ Paused.';
    }}
    function resumeText() {{
        window.speechSynthesis.resume();
        document.getElementById('wave').style.display = 'flex';
        document.getElementById('statusMsg').innerText = '▶️ Resumed.';
    }}

    // Autoplay if requested
    {'speakText();' if autoplay else ''}
    </script>
    """


def build_voice_script(patient_name, age, gender, bmi, blood_pressure,
                        blood_sugar, cholesterol, heart_rate, spo2,
                        creatinine, smoker, diabetes_history, family_history,
                        risk_name, confidence, recs, xgb_name, fed_name):
    """Build a natural language voice script for the patient."""
    rec_text = ". ".join([f"Recommendation {i+1}: {r}" for i, r in enumerate(recs)])

    risk_advice = {
        "LOW RISK":    "You are in good health. Keep maintaining your healthy lifestyle and get regular checkups.",
        "MEDIUM RISK": "You have some health concerns that need attention. Please follow the recommendations and consult a doctor soon.",
        "HIGH RISK":   "You are at high risk. Please seek immediate medical attention. Do not ignore these warning signs."
    }

    script = f"""
Hello {patient_name}. Welcome to FedHealth Twin, your AI-based smart health risk prediction system.

I have completed the analysis of your health data. Here is your complete health report.

Patient Details:
Your age is {age} years. You are {gender}.
Your Body Mass Index, or BMI, is {bmi}.
Your blood pressure is {blood_pressure} millimeters of mercury.
Your blood sugar level is {blood_sugar} milligrams per deciliter.
Your cholesterol is {cholesterol} milligrams per deciliter.
Your heart rate is {heart_rate} beats per minute.
Your oxygen saturation, or SpO2, is {spo2} percent.
Your creatinine level is {creatinine} milligrams per deciliter.
{'You are a smoker. This significantly increases your health risk.' if smoker else 'You are a non-smoker. This is good for your health.'}
{'You have a history of diabetes.' if diabetes_history else 'You have no diabetes history.'}
{'You have a family history of disease.' if family_history else 'You have no significant family history of disease.'}

AI Prediction Results:
The XGBoost machine learning model predicts you are at {xgb_name}.
The Federated Learning model predicts you are at {fed_name}.
After combining both models, your final health risk prediction is:

{risk_name}.

The AI system is {confidence:.1f} percent confident in this prediction.

{risk_advice.get(risk_name, '')}

Personalized Recommendations:
{rec_text}.

Important Disclaimer:
This report is generated by an artificial intelligence system for informational purposes only.
Please always consult a qualified medical professional for proper diagnosis and treatment.

Thank you for using FedHealth Twin. Stay healthy and take care of yourself.
    """.strip()
    return script


# ─── Load Models ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    import sys
    sys.path.append('.')
    from federated_learning import FederatedNet
    with open('models/xgboost_model.pkl','rb') as f:
        xgb = pickle.load(f)
    with open('models/scaler.pkl','rb') as f:
        scaler = pickle.load(f)
    fed = FederatedNet(input_dim=12, num_classes=3)
    fed.load_state_dict(torch.load('models/federated_model.pth',
                                    map_location=torch.device('cpu')))
    fed.eval()
    return xgb, scaler, fed

try:
    xgb_model, scaler, fed_model = load_models()
    models_loaded = True
except Exception as e:
    models_loaded  = False
    model_error    = str(e)


# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div style="font-size:2.8rem;">❤️</div>
        <h2>FedHealth-Twin</h2>
        <p>AI Smart Health Prediction</p>
        <p>🔒 Privacy Preserved | 🧠 AI Powered</p>
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

    st.markdown("### 🏥 Medical History")
    smoker           = st.checkbox("🚬 Smoker")
    diabetes_history = st.checkbox("🩸 Diabetes History")
    family_history   = st.checkbox("👨‍👩‍👧 Family History of Disease")

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Health Risk", use_container_width=True)
    st.markdown("""
    <div style="text-align:center;margin-top:14px;color:#555;font-size:0.7rem;">
        🔒 Data never leaves your device<br>Powered by Federated Learning
    </div>""", unsafe_allow_html=True)


# ─── Hero Banner ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-icon-row">
    <div class="hero-icon icon-red">💓</div>
    <div class="hero-icon icon-blue">🧠</div>
    <div class="hero-icon icon-green">🛡️</div>
    <div class="hero-icon icon-purple">📊</div>
    <div class="hero-icon icon-cyan">🎙️</div>
  </div>
  <div class="hero-title">🏥 FedHealth-Twin</div>
  <div class="hero-subtitle">AI-Based Smart Health Risk Prediction System</div>
  <div class="hero-badges">
    <span class="badge badge-blue">🔒 Federated Learning</span>
    <span class="badge badge-red">🧠 Deep Learning</span>
    <span class="badge badge-green">📊 Explainable AI</span>
    <span class="badge badge-purple">🔬 Multimodal Fusion</span>
    <span class="badge badge-orange">🎙️ Voice Assistant</span>
  </div>
</div>
""", unsafe_allow_html=True)


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


# ─── Main ─────────────────────────────────────────────────────
if predict_btn:
    if not models_loaded:
        st.error(f"❌ Models not loaded: {model_error}")
        st.stop()

    patient_data = {
        'age': age, 'gender': 1 if gender=="Male" else 0,
        'bmi': bmi, 'blood_pressure': blood_pressure,
        'blood_sugar': blood_sugar, 'cholesterol': cholesterol,
        'heart_rate': heart_rate, 'spo2': spo2, 'creatinine': creatinine,
        'smoker': 1 if smoker else 0,
        'diabetes_history': 1 if diabetes_history else 0,
        'family_history':   1 if family_history   else 0
    }

    with st.spinner("🔄 Analysing with AI models..."):
        fcols = ['age','gender','bmi','blood_pressure','blood_sugar',
                 'cholesterol','heart_rate','spo2','creatinine',
                 'smoker','diabetes_history','family_history']
        pdf       = pd.DataFrame([patient_data])[fcols]
        xgb_pred  = xgb_model.predict(pdf)[0]
        xgb_proba = xgb_model.predict_proba(pdf)[0]

        ps = scaler.transform(pdf)
        with torch.no_grad():
            fo        = fed_model(torch.FloatTensor(ps))
            fed_proba = torch.softmax(fo, dim=1).numpy()[0]
        fed_pred = np.argmax(fed_proba)

        ens_proba  = (xgb_proba + fed_proba) / 2
        final_pred = np.argmax(ens_proba)
        confidence = ens_proba[final_pred] * 100

    risk_labels = ['Low Risk ✅','Medium Risk ⚠️','High Risk 🚨']
    risk_css    = ['risk-low','risk-medium','risk-high']
    risk_names  = ['LOW RISK','MEDIUM RISK','HIGH RISK']

    # Build recommendations
    if final_pred == 0:
        recs = [
            "Continue regular exercise for 30 minutes daily",
            "Maintain a balanced diet rich in fruits and vegetables",
            "Get regular health checkups every 6 months",
            "Stay hydrated and maintain a healthy sleep schedule"
        ]
    elif final_pred == 1:
        recs = []
        if bmi > 28:             recs.append("Work on weight reduction, target BMI below 25")
        if blood_pressure > 120: recs.append("Monitor blood pressure daily and reduce salt intake")
        if blood_sugar > 150:    recs.append("Control carbohydrate intake and monitor sugar weekly")
        if smoker:               recs.append("Quit smoking as it is a major risk amplifier")
        recs.append("Exercise 5 days per week for 45 minutes")
        recs.append("Consult a doctor within the next 30 days")
    else:
        recs = []
        if bmi > 30:             recs.append("Urgently start a medically supervised weight loss program")
        if blood_pressure > 140: recs.append("Urgently seek treatment for high blood pressure")
        if blood_sugar > 250:    recs.append("Urgently consult an endocrinologist for diabetes management")
        if spo2 < 92:            recs.append("Urgently seek emergency evaluation for low oxygen levels")
        if creatinine > 2.5:     recs.append("Urgently get a kidney function evaluation")
        if smoker:               recs.append("Stop smoking immediately")
        recs.append("Seek hospitalization or specialist consultation right away")

    # ── Voice Assistant FIRST ────────────────────────────────
    st.markdown('<div class="section-header">🎙️ AI Voice Assistant</div>',
                unsafe_allow_html=True)

    voice_script = build_voice_script(
        patient_name, age, gender, bmi, blood_pressure,
        blood_sugar, cholesterol, heart_rate, spo2,
        creatinine, smoker, diabetes_history, family_history,
        risk_names[final_pred], confidence, recs,
        risk_names[xgb_pred], risk_names[fed_pred]
    )
    st.components.v1.html(
        voice_assistant_js(voice_script, autoplay=False),
        height=260
    )

    st.markdown("---")

    # ── Result ────────────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 Prediction Result</div>',
                unsafe_allow_html=True)
    c1,c2,c3 = st.columns([2,1,1])
    with c1:
        st.markdown(f"""
        <div class="{risk_css[final_pred]}">
            {risk_labels[final_pred]}<br>
            <small style="font-size:.9rem;font-weight:400;">Confidence: {confidence:.1f}%</small><br>
            <small style="font-size:.8rem;font-weight:300;">Patient: {patient_name}</small>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="info-card">
            <h3>XGBoost</h3>
            <p>{risk_names[xgb_pred]}</p>
            <p style="color:#1a73e8;font-weight:600;">{max(xgb_proba)*100:.1f}%</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="info-card">
            <h3>FedModel</h3>
            <p>{risk_names[fed_pred]}</p>
            <p style="color:#ff4b6e;font-weight:600;">{max(fed_proba)*100:.1f}%</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Risk Probability Distribution</div>',
                unsafe_allow_html=True)
    colors  = ['#2e7d32','#e64a19','#c62828']
    rlabels = ['Low Risk','Medium Risk','High Risk']
    c1,c2   = st.columns(2)

    with c1:
        fig,ax = plt.subplots(figsize=(5,4), facecolor='#16213e')
        ax.set_facecolor('#16213e')
        bars = ax.bar(rlabels, ens_proba*100, color=colors,
                      edgecolor='#ffffff11', linewidth=1, width=0.45)
        for bar,val in zip(bars, ens_proba):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                    f'{val*100:.1f}%', ha='center',
                    fontweight='bold', color='white', fontsize=10)
        ax.set_ylabel("Probability (%)", color='#aaa', fontsize=9)
        ax.set_title("Risk Probability", color='white', fontweight='bold', fontsize=11, pad=10)
        ax.set_ylim(0,115)
        ax.tick_params(colors='#aaa', labelsize=8)
        for sp in ['top','right']:   ax.spines[sp].set_visible(False)
        for sp in ['bottom','left']: ax.spines[sp].set_color('#333')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c2:
        fig,ax = plt.subplots(figsize=(5,4), facecolor='#16213e')
        ax.set_facecolor('#16213e')
        wedges,texts,autotexts = ax.pie(
            ens_proba, labels=rlabels, colors=colors,
            autopct='%1.1f%%', startangle=90,
            wedgeprops=dict(edgecolor='#16213e', linewidth=2)
        )
        for t in texts:     t.set_color('#ccc'); t.set_fontsize(9)
        for t in autotexts: t.set_color('white'); t.set_fontweight('bold'); t.set_fontsize(9)
        ax.set_title("Risk Distribution", color='white', fontweight='bold', fontsize=11, pad=10)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")

    # ── Vitals ────────────────────────────────────────────────
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
        ("Heart Rate",  heart_rate,  " bpm",    60, 100, "Normal: 60–100",  True),
        ("Cholesterol", cholesterol, " mg/dL", 100, 200, "Normal: <200",    True),
        ("Creatinine",  creatinine,  " mg/dL", 0.5, 1.2, "Normal: 0.5–1.2", True),
        ("Age",         age,         " yrs",     0, 200, "",                True),
    ]):
        col.markdown(vital_card(*args), unsafe_allow_html=True)

    st.markdown("---")

    # ── SHAP ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔍 Explainable AI — SHAP Analysis</div>',
                unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        try:    st.image('reports/shap_global.png', caption="🌍 Global Feature Importance", use_container_width=True)
        except: st.info("Run explainability.py to generate SHAP plots")
    with c2:
        try:    st.image('reports/shap_patient.png', caption="👤 Patient-Level Explanation", use_container_width=True)
        except: st.info("SHAP patient plot not available")

    st.markdown("---")

    # ── Recommendations ───────────────────────────────────────
    st.markdown('<div class="section-header">💡 Personalized Recommendations</div>',
                unsafe_allow_html=True)
    if final_pred == 0: st.success("✅ LOW RISK — Keep maintaining your healthy lifestyle!")
    elif final_pred==1: st.warning("⚠️ MEDIUM RISK — Lifestyle changes recommended!")
    else:               st.error("🚨 HIGH RISK — Immediate medical attention required!")
    for i,rec in enumerate(recs,1):
        st.markdown(f"**{i}.** {rec}")

    st.markdown("---")

    # ── Report Download ───────────────────────────────────────
    st.markdown('<div class="section-header">📄 Health Report</div>',
                unsafe_allow_html=True)
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""FedHealth-Twin: AI Health Risk Report
Generated: {now} | Patient: {patient_name}
{'='*60}
VITALS: Age:{age} | BMI:{bmi} | BP:{blood_pressure} | Sugar:{blood_sugar}
HR:{heart_rate} | SpO2:{spo2}% | Cholesterol:{cholesterol} | Creatinine:{creatinine}
Smoker:{'Yes' if smoker else 'No'} | Diabetes Hx:{'Yes' if diabetes_history else 'No'} | Family Hx:{'Yes' if family_history else 'No'}
{'='*60}
RESULT: {risk_names[final_pred]} — Confidence: {confidence:.1f}%
XGBoost: {risk_names[xgb_pred]} | FedModel: {risk_names[fed_pred]}
{'='*60}
RECOMMENDATIONS:
""" + "\n".join([f"  {i+1}. {r}" for i,r in enumerate(recs)]) + \
"\n\nDISCLAIMER: AI-generated. Always consult a medical professional.\n"

    st.text_area("Full Report", report, height=220)
    st.download_button("⬇️ Download Full Report", data=report,
                       file_name=f"{patient_name}_health_report.txt",
                       mime="text/plain")

    st.markdown("---")

    # ── Privacy ───────────────────────────────────────────────
    st.markdown('<div class="section-header">🔒 Federated Learning — Privacy Architecture</div>',
                unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    for col,name in zip([c1,c2,c3],["🏥 Hospital A","🏥 Hospital B","🏥 Hospital C"]):
        col.markdown(f'<div class="info-card"><h3>{name}</h3><p>Local Training Only<br>Data stays private</p></div>',
                     unsafe_allow_html=True)
    st.success("🔒 Only model weights shared via FedAvg — Patient data NEVER leaves local systems!")


# ─── Welcome Screen ──────────────────────────────────────────
else:
    c1,c2,c3,c4 = st.columns(4)
    for col,(icon,title,desc) in zip([c1,c2,c3,c4],[
        ("🔒","Privacy First","Federated Learning — data never leaves device"),
        ("🧠","AI Powered","LSTM + XGBoost + Fusion Neural Network"),
        ("📊","Explainable AI","SHAP analysis for every prediction"),
        ("🎙️","Voice Assistant","Speaks your full health report aloud"),
    ]):
        col.markdown(f"""
        <div class="feature-card">
            <div style="font-size:2.4rem;">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📈 System Statistics</div>',
                unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    for col,(num,label) in zip([c1,c2,c3,c4],[
        ("3","AI Models Combined"),("12","Health Features"),
        ("3","Hospital Nodes"),("🎙️","Voice Enabled"),
    ]):
        col.markdown(f'<div class="info-card"><h3>{num}</h3><p>{label}</p></div>',
                     unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">🚀 How to Use</div>',
                unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Step 1** — Enter patient name and demographics in the sidebar  
        **Step 2** — Adjust clinical vitals using the sliders  
        **Step 3** — Tick relevant medical history checkboxes  
        **Step 4** — Click **🔍 Predict Health Risk**  
        **Step 5** — Click **🔊 Speak Health Report** to hear results  
        **Step 6** — Download the full health report  
        """)
    with c2:
        st.markdown("""
        | Feature | Technology |
        |---------|-----------|
        | Tabular Model | XGBoost |
        | Time-Series | PyTorch LSTM |
        | Privacy | Federated Learning |
        | Explainability | SHAP |
        | **Voice Assistant** | **Web Speech API** |
        | Dashboard | Streamlit |
        """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#444;padding:16px;font-size:0.85rem;">
        ❤️ FedHealth-Twin &nbsp;|&nbsp; AI-Based Smart Health Risk Prediction System
    </div>""", unsafe_allow_html=True)
