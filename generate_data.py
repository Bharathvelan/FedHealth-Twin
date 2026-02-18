import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

data = {
    'age': np.random.randint(20, 80, n),
    'gender': np.random.choice([0, 1], n),
    'bmi': np.round(np.random.uniform(15, 45, n), 1),
    'blood_pressure': np.random.randint(60, 180, n),
    'blood_sugar': np.random.randint(70, 400, n),
    'cholesterol': np.random.randint(100, 350, n),
    'heart_rate': np.random.randint(50, 120, n),
    'spo2': np.round(np.random.uniform(85, 100, n), 1),
    'creatinine': np.round(np.random.uniform(0.5, 5.0, n), 2),
    'smoker': np.random.choice([0, 1], n),
    'diabetes_history': np.random.choice([0, 1], n),
    'family_history': np.random.choice([0, 1], n),
}

df = pd.DataFrame(data)

# Create risk label based on conditions
def calculate_risk(row):
    score = 0
    if row['age'] > 55: score += 2
    if row['bmi'] > 30: score += 2
    if row['blood_pressure'] > 130: score += 2
    if row['blood_sugar'] > 200: score += 3
    if row['cholesterol'] > 240: score += 2
    if row['spo2'] < 94: score += 3
    if row['smoker'] == 1: score += 2
    if row['diabetes_history'] == 1: score += 2
    if row['family_history'] == 1: score += 1
    if row['creatinine'] > 2.0: score += 2
    
    if score <= 4: return 0   # Low Risk
    elif score <= 9: return 1  # Medium Risk
    else: return 2             # High Risk

df['risk_label'] = df.apply(calculate_risk, axis=1)

df.to_csv('data/health_data.csv', index=False)
print("✅ Dataset created successfully!")
print(df['risk_label'].value_counts())
print(df.head())
