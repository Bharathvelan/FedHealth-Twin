import pandas as pd
import numpy as np
import pickle
import torch
from datetime import datetime
from federated_learning import FederatedNet

# ─── Load Models ─────────────────────────────────────────────
with open('models/xgboost_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

fed_model = FederatedNet(input_dim=12, num_classes=3)
fed_model.load_state_dict(torch.load('models/federated_model.pth'))
fed_model.eval()

# ─── Risk Labels ─────────────────────────────────────────────
risk_names  = {0: 'LOW RISK', 1: 'MEDIUM RISK', 2: 'HIGH RISK'}
risk_emojis = {0: '✅', 1: '⚠️', 2: '🚨'}

risk_descriptions = {
    0: "The patient shows no significant health risk indicators. "
       "Current lifestyle and vitals are within healthy ranges.",
    1: "The patient shows moderate health risk. "
       "Some indicators are borderline and require attention and lifestyle changes.",
    2: "The patient is at HIGH risk. "
       "Immediate medical consultation is strongly recommended. "
       "Multiple critical indicators are outside safe ranges."
}

# ─── Lifestyle Recommendations ────────────────────────────────
def get_recommendations(patient, risk_level):
    recs = []

    if risk_level == 0:
        recs = [
            "Continue regular exercise (30 min daily)",
            "Maintain balanced diet rich in fruits and vegetables",
            "Regular health checkups every 6 months",
            "Stay hydrated and maintain healthy sleep schedule"
        ]
    elif risk_level == 1:
        if patient['bmi'] > 28:
            recs.append("Work on weight reduction — target BMI below 25")
        if patient['blood_pressure'] > 120:
            recs.append("Monitor blood pressure daily, reduce salt intake")
        if patient['blood_sugar'] > 150:
            recs.append("Control carbohydrate intake, monitor blood sugar weekly")
        if patient['smoker'] == 1:
            recs.append("Quit smoking immediately — major risk amplifier")
        recs.append("Exercise at least 5 days a week, 45 minutes per session")
        recs.append("Consult a doctor within next 30 days for full checkup")
    else:
        if patient['bmi'] > 30:
            recs.append("URGENT: Start medically supervised weight loss program")
        if patient['blood_pressure'] > 140:
            recs.append("URGENT: Seek immediate treatment for hypertension")
        if patient['blood_sugar'] > 250:
            recs.append("URGENT: Consult endocrinologist for diabetes management")
        if patient['spo2'] < 92:
            recs.append("URGENT: Low SpO2 — seek emergency medical evaluation")
        if patient['creatinine'] > 2.5:
            recs.append("URGENT: High creatinine — kidney function evaluation needed")
        if patient['smoker'] == 1:
            recs.append("URGENT: Stop smoking immediately")
        recs.append("Hospitalization or immediate specialist consultation advised")

    return recs

# ─── Generate Report Function ─────────────────────────────────
def generate_report(patient_data, patient_name="Patient"):
    # Prepare input
    feature_cols = ['age', 'gender', 'bmi', 'blood_pressure', 'blood_sugar',
                    'cholesterol', 'heart_rate', 'spo2', 'creatinine',
                    'smoker', 'diabetes_history', 'family_history']

    patient_df = pd.DataFrame([patient_data])[feature_cols]

    # XGBoost prediction
    xgb_pred  = xgb_model.predict(patient_df)[0]
    xgb_proba = xgb_model.predict_proba(patient_df)[0]

    # Federated model prediction
    patient_scaled = scaler.transform(patient_df)
    patient_tensor = torch.FloatTensor(patient_scaled)
    with torch.no_grad():
        fed_out   = fed_model(patient_tensor)
        fed_proba = torch.softmax(fed_out, dim=1).numpy()[0]
        fed_pred  = np.argmax(fed_proba)

    # Ensemble final prediction
    ensemble_proba = (xgb_proba + fed_proba) / 2
    final_pred     = np.argmax(ensemble_proba)
    confidence     = ensemble_proba[final_pred] * 100

    # Recommendations
    recs = get_recommendations(patient_data, final_pred)

    # ─── Build Report ─────────────────────────────────────────
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║         FedHealth-Twin: AI Health Risk Report                ║
╚══════════════════════════════════════════════════════════════╝

Report Generated : {now}
Patient Name     : {patient_name}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATIENT VITALS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Age              : {patient_data['age']} years
Gender           : {'Male' if patient_data['gender']==1 else 'Female'}
BMI              : {patient_data['bmi']}
Blood Pressure   : {patient_data['blood_pressure']} mmHg
Blood Sugar      : {patient_data['blood_sugar']} mg/dL
Cholesterol      : {patient_data['cholesterol']} mg/dL
Heart Rate       : {patient_data['heart_rate']} bpm
SpO2             : {patient_data['spo2']}%
Creatinine       : {patient_data['creatinine']} mg/dL
Smoker           : {'Yes' if patient_data['smoker']==1 else 'No'}
Diabetes History : {'Yes' if patient_data['diabetes_history']==1 else 'No'}
Family History   : {'Yes' if patient_data['family_history']==1 else 'No'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PREDICTION RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
XGBoost Model    : {risk_names[xgb_pred]} 
                   (Low:{xgb_proba[0]*100:.1f}% | 
                    Med:{xgb_proba[1]*100:.1f}% | 
                    High:{xgb_proba[2]*100:.1f}%)

Federated Model  : {risk_names[fed_pred]}
                   (Low:{fed_proba[0]*100:.1f}% | 
                    Med:{fed_proba[1]*100:.1f}% | 
                    High:{fed_proba[2]*100:.1f}%)

FINAL PREDICTION : {risk_emojis[final_pred]} {risk_names[final_pred]}
Confidence Score : {confidence:.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HEALTH SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{risk_descriptions[final_pred]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PERSONALIZED RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

    for i, rec in enumerate(recs, 1):
        report += f"\n  {i}. {rec}"

    report += """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISCLAIMER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This report is generated by an AI system for informational
purposes only. Always consult a qualified medical professional
for diagnosis and treatment decisions.
══════════════════════════════════════════════════════════════
"""
    return report, final_pred, confidence, ensemble_proba

# ─── Test with Sample Patient ─────────────────────────────────
sample_patient = {
    'age': 55,
    'gender': 1,
    'bmi': 34.5,
    'blood_pressure': 155,
    'blood_sugar': 280,
    'cholesterol': 260,
    'heart_rate': 95,
    'spo2': 91.5,
    'creatinine': 2.8,
    'smoker': 1,
    'diabetes_history': 1,
    'family_history': 1
}

report, pred, conf, proba = generate_report(sample_patient, "Test Patient")
print(report)

# Save report
with open('reports/health_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("Report saved to reports/health_report.txt")

