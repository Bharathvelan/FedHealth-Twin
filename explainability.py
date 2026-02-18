import numpy as np
import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─── Load XGBoost Model and Test Data ────────────────────────
with open('models/xgboost_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

df = pd.read_csv('data/health_data.csv')
X = df.drop('risk_label', axis=1)
y = df['risk_label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─── SHAP Explainer ───────────────────────────────────────────
print("🔄 Generating SHAP Explanations...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# ─── Plot 1: Global Feature Importance ───────────────────────
print("📊 Saving Global Feature Importance plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values, 
    X_test, 
    class_names=['Low Risk', 'Medium Risk', 'High Risk'],
    show=False
)
plt.title("Global Feature Importance (SHAP)", fontsize=14)
plt.tight_layout()
plt.savefig('reports/shap_global.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: reports/shap_global.png")

# ─── Plot 2: Single Patient Explanation ──────────────────────
print("📊 Saving Single Patient Explanation plot...")
patient_idx = 0  # First test patient

# Get patient details
patient = X_test.iloc[patient_idx]
true_risk = y_test.iloc[patient_idx]
pred_risk = xgb_model.predict(X_test.iloc[[patient_idx]])[0]

risk_names = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}

print(f"\n👤 Patient Details:")
print(f"   Age: {patient['age']}")
print(f"   BMI: {patient['bmi']}")
print(f"   Blood Pressure: {patient['blood_pressure']}")
print(f"   Blood Sugar: {patient['blood_sugar']}")
print(f"   SpO2: {patient['spo2']}")
print(f"   True Risk: {risk_names[true_risk]}")
print(f"   Predicted Risk: {risk_names[pred_risk]}")

# SHAP waterfall for single patient (High risk class = 2)
plt.figure(figsize=(10, 6))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[pred_risk][patient_idx],
        base_values=explainer.expected_value[pred_risk],
        data=X_test.iloc[patient_idx],
        feature_names=X_test.columns.tolist()
    ),
    show=False
)
plt.title(f"Patient Explanation - Predicted: {risk_names[pred_risk]}", fontsize=13)
plt.tight_layout()
plt.savefig('reports/shap_patient.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: reports/shap_patient.png")

# ─── Counterfactual Analysis ──────────────────────────────────
print("\n🔄 Generating Counterfactual Suggestions...")

def generate_counterfactual(patient, model):
    suggestions = []
    patient_copy = patient.copy()
    pred = model.predict(pd.DataFrame([patient_copy]))[0]

    if pred >= 1:  # Medium or High risk
        checks = {
            'bmi':            (30,  'Reduce BMI below 30'),
            'blood_pressure': (130, 'Reduce Blood Pressure below 130'),
            'blood_sugar':    (200, 'Reduce Blood Sugar below 200'),
            'cholesterol':    (240, 'Reduce Cholesterol below 240'),
            'spo2':           (94,  'Improve SpO2 above 94%'),
            'creatinine':     (2.0, 'Reduce Creatinine below 2.0'),
        }
        thresholds_high = ['bmi', 'blood_pressure', 'blood_sugar', 
                           'cholesterol', 'creatinine']

        for feature, (threshold, message) in checks.items():
            val = patient_copy[feature]
            if feature in thresholds_high and val > threshold:
                suggestions.append(f"⚠️  {message} (Current: {val})")
            elif feature == 'spo2' and val < threshold:
                suggestions.append(f"⚠️  {message} (Current: {val})")

        if patient_copy['smoker'] == 1:
            suggestions.append("⚠️  Quit smoking to reduce risk significantly")
        
        if not suggestions:
            suggestions.append("✅ Vitals are borderline — maintain healthy lifestyle")

    else:
        suggestions.append("✅ Patient is at Low Risk. Maintain current lifestyle!")

    return suggestions

# Test on first patient
suggestions = generate_counterfactual(patient, xgb_model)
print(f"\n💡 Counterfactual Suggestions for Patient:")
for s in suggestions:
    print(f"   {s}")

# Save suggestions
with open('reports/counterfactual.txt', 'w', encoding='utf-8') as f:
    f.write("COUNTERFACTUAL SUGGESTIONS\n")
    f.write("="*40 + "\n")
    for s in suggestions:
        f.write(s + "\n")

print("\n✅ Counterfactual saved to reports/counterfactual.txt")
print("✅ Explainability module complete!")

