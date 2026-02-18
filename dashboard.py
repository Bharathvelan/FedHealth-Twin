import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
import shap
from report_generator import generate_report
from federated_learning import FederatedNet

# ─── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="FedHealth-Twin",
    page_icon="🏥",
    layout="wide"
)

# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f0f4f8; }
    .title {
        text-align: center;
        color: #1a73e8;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1.1rem;
    }
    .risk-low {
        background-color: #d4edda;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #155724;
    }
    .risk-medium {
        background-color: #fff3cd;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #856404;
    }
    .risk-high {
        background-color: #f8d7da;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────
st.markdown('<p class="title">🏥 FedHealth-Twin</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Based Smart Health Risk Prediction System</p>',
            unsafe_allow_html=True)
st.markdown('<p class="subtitle">Federated Learning | Explainable AI | Multimodal Fusion</p>',
            unsafe_allow_html=True)
st.markdown("---")

# ─── Sidebar ──────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/000000/heart-with-pulse.png", width=80)
st.sidebar.title("Patient Input")
st.sidebar.markdown("Enter patient details below:")

patient_name = st.sidebar.text_input("Patient Name", value="John Doe")

st.sidebar.markdown("### 👤 Demographics")
age    = st.sidebar.slider("Age", 18, 90, 45)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

st.sidebar.markdown("### 📊 Clinical Vitals")
bmi            = st.sidebar.slider("BMI", 15.0, 50.0, 25.0, 0.1)
blood_pressure = st.sidebar.slider("Blood Pressure (mmHg)", 60, 200, 120)
blood_sugar    = st.sidebar.slider("Blood Sugar (mg/dL)", 70, 500, 100)
cholesterol    = st.sidebar.slider("Cholesterol (mg/dL)", 100, 400, 180)
heart_rate     = st.sidebar.slider("Heart Rate (bpm)", 40, 150, 75)
spo2           = st.sidebar.slider("SpO2 (%)", 80.0, 100.0, 98.0, 0.1)
creatinine     = st.sidebar.slider("Creatinine (mg/dL)", 0.5, 6.0, 1.0, 0.1)

st.sidebar.markdown("### 🏥 Medical History")
smoker           = st.sidebar.checkbox("Smoker")
diabetes_history = st.sidebar.checkbox("Diabetes History")
family_history   = st.sidebar.checkbox("Family History of Disease")

predict_btn = st.sidebar.button("🔍 Predict Health Risk", use_container_width=True)

# ─── Main Dashboard ───────────────────────────────────────────
if predict_btn:
    patient_data = {
        'age': age,
        'gender': 1 if gender == "Male" else 0,
        'bmi': bmi,
        'blood_pressure': blood_pressure,
        'blood_sugar': blood_sugar,
        'cholesterol': cholesterol,
        'heart_rate': heart_rate,
        'spo2': spo2,
        'creatinine': creatinine,
        'smoker': 1 if smoker else 0,
        'diabetes_history': 1 if diabetes_history else 0,
        'family_history': 1 if family_history else 0
    }

    with st.spinner("🔄 Analyzing patient data..."):
        report, final_pred, confidence, ensemble_proba = generate_report(
            patient_data, patient_name
        )

    # ─── Risk Result ──────────────────────────────────────────
    st.markdown("## 🎯 Prediction Result")
    col1, col2, col3 = st.columns(3)

    risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']
    risk_colors = ['risk-low', 'risk-medium', 'risk-high']
    risk_icons  = ['✅', '⚠️', '🚨']

    with col1:
        st.markdown(f"""
        <div class="{risk_colors[final_pred]}">
            {risk_icons[final_pred]} {risk_labels[final_pred]}<br>
            <small>Confidence: {confidence:.1f}%</small>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.metric("XGBoost Prediction",
                  risk_labels[np.argmax(ensemble_proba)],
                  f"{max(ensemble_proba)*100:.1f}% confidence")

    with col3:
        st.metric("Federated Model",
                  risk_labels[final_pred],
                  f"{confidence:.1f}% confidence")

    st.markdown("---")

    # ─── Risk Probability Chart ───────────────────────────────
    st.markdown("## 📊 Risk Probability Distribution")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#28a745', '#ffc107', '#dc3545']
        bars = ax.bar(risk_labels, ensemble_proba * 100, color=colors, 
                      edgecolor='white', linewidth=1.5)
        ax.set_ylabel("Probability (%)")
        ax.set_title("Health Risk Probability")
        ax.set_ylim(0, 100)
        for bar, val in zip(bars, ensemble_proba):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    f'{val*100:.1f}%', ha='center', fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        wedges, texts, autotexts = ax.pie(
            ensemble_proba,
            labels=risk_labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90
        )
        ax.set_title("Risk Distribution (Pie Chart)")
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ─── Patient Vitals Summary ───────────────────────────────
    st.markdown("## 📋 Patient Vitals Summary")
    col1, col2, col3, col4 = st.columns(4)

    def status(val, low, high):
        if val < low or val > high:
            return "🔴"
        return "🟢"

    col1.metric("BMI", f"{bmi}",
                status(bmi, 18.5, 25) + " Normal: 18.5-25")
    col2.metric("Blood Pressure", f"{blood_pressure} mmHg",
                status(blood_pressure, 60, 120) + " Normal: <120")
    col3.metric("Blood Sugar", f"{blood_sugar} mg/dL",
                status(blood_sugar, 70, 140) + " Normal: 70-140")
    col4.metric("SpO2", f"{spo2}%",
                status(spo2, 95, 100) + " Normal: >95%")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Heart Rate", f"{heart_rate} bpm",
                status(heart_rate, 60, 100) + " Normal: 60-100")
    col2.metric("Cholesterol", f"{cholesterol} mg/dL",
                status(cholesterol, 100, 200) + " Normal: <200")
    col3.metric("Creatinine", f"{creatinine} mg/dL",
                status(creatinine, 0.5, 1.2) + " Normal: 0.5-1.2")
    col4.metric("Age", f"{age} years", "")

    st.markdown("---")

    # ─── SHAP Explanation ─────────────────────────────────────
    st.markdown("## 🔍 Explainable AI (SHAP Analysis)")
    col1, col2 = st.columns(2)

    with col1:
        try:
            st.image('reports/shap_global.png',
                     caption="Global Feature Importance",
                     use_container_width=True)
        except:
            st.info("Run explainability.py first to generate SHAP plots")

    with col2:
        try:
            st.image('reports/shap_patient.png',
                     caption="Patient-Level Explanation",
                     use_container_width=True)
        except:
            st.info("SHAP patient plot not available")

    st.markdown("---")

    # ─── Recommendations ──────────────────────────────────────
    st.markdown("## 💡 Personalized Recommendations")

    if final_pred == 0:
        st.success("✅ Patient is at LOW RISK. Keep maintaining healthy habits!")
    elif final_pred == 1:
        st.warning("⚠️ Patient is at MEDIUM RISK. Lifestyle changes recommended.")
    else:
        st.error("🚨 Patient is at HIGH RISK. Immediate medical attention required!")

    # Extract recommendations from report
    lines = report.split('\n')
    in_recs = False
    for line in lines:
        if 'PERSONALIZED RECOMMENDATIONS' in line:
            in_recs = True
            continue
        if in_recs and line.strip().startswith(tuple('123456789')):
            st.markdown(f"- {line.strip()}")
        if in_recs and '━━━' in line and line != lines[lines.index(line)-1]:
            if any(c.isdigit() for c in ''.join(lines[lines.index(line)-5:lines.index(line)])):
                in_recs = False

    st.markdown("---")

    # ─── Full Report Download ─────────────────────────────────
    st.markdown("## 📄 Full Health Report")
    st.text_area("Complete Report", report, height=300)
    st.download_button(
        label="⬇️ Download Full Report",
        data=report,
        file_name=f"{patient_name}_health_report.txt",
        mime="text/plain"
    )

    st.markdown("---")

    # ─── Federated Learning Info ──────────────────────────────
    st.markdown("## 🔒 Privacy & Federated Learning")
    col1, col2, col3 = st.columns(3)
    col1.info("🏥 Hospital A\nLocal Training\nData stays private")
    col2.info("🏥 Hospital B\nLocal Training\nData stays private")
    col3.info("🏥 Hospital C\nLocal Training\nData stays private")
    st.success("🔒 Only model weights are shared — Patient data NEVER leaves local systems!")

else:
    # ─── Welcome Screen ───────────────────────────────────────
    st.markdown("## 👋 Welcome to FedHealth-Twin")

    col1, col2, col3 = st.columns(3)
    col1.success("🔒 **Privacy First**\nFederated Learning ensures patient data never leaves local systems")
    col2.info("🧠 **AI Powered**\nMultimodal fusion of tabular, time-series and text data")
    col3.warning("📊 **Explainable**\nSHAP analysis explains every prediction transparently")

    st.markdown("---")
    st.markdown("### 🚀 How to Use")
    st.markdown("""
    1. Enter patient details in the **left sidebar**
    2. Click **Predict Health Risk** button
    3. View risk prediction, charts, and SHAP explanations
    4. Download the full health report
    """)

    st.markdown("---")
    st.markdown("### 🏗️ System Architecture")
    st.markdown("""
    | Module | Technology |
    |--------|-----------|
    | Tabular Data Model | XGBoost + Calibration |
    | Time-Series Model | PyTorch LSTM |
    | Text Encoder | TF-IDF + SVD |
    | Fusion Layer | PyTorch Neural Network |
    | Privacy | Federated Learning (FedAvg) |
    | Explainability | SHAP + Counterfactual Analysis |
    | Dashboard | Streamlit |
    """)

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:gray;'>FedHealth-Twin | "
        "Final Year Project | AI & ML | 2025</p>",
        unsafe_allow_html=True
    )
