import pandas as pd
import numpy as np

np.random.seed(42)
n = 2000

data = {
    'age':            np.random.randint(20, 80, n),
    'gender':         np.random.choice([0, 1], n),
    'bmi':            np.round(np.random.uniform(15, 45, n), 1),
    'blood_pressure': np.random.randint(60, 200, n),
    'blood_sugar':    np.random.randint(70, 400, n),
    'cholesterol':    np.random.randint(100, 350, n),
    'heart_rate':     np.random.randint(50, 130, n),
    'spo2':           np.round(np.random.uniform(85, 100, n), 1),
    'creatinine':     np.round(np.random.uniform(0.5, 6.0, n), 2),
    'hba1c':          np.round(np.random.uniform(4.0, 12.0, n), 1),
    'triglycerides':  np.random.randint(50, 500, n),
    'alt_enzyme':     np.random.randint(10, 200, n),
    'smoker':         np.random.choice([0, 1], n),
    'diabetes_history':  np.random.choice([0, 1], n),
    'family_history':    np.random.choice([0, 1], n),
    'alcohol':           np.random.choice([0, 1], n),
    'physical_activity': np.random.choice([0, 1, 2], n),
}

df = pd.DataFrame(data)

# ── Disease Labels ─────────────────────────────────────────

# 1. Diabetes Risk
def diabetes_risk(row):
    score = 0
    if row['blood_sugar'] > 200:  score += 3
    if row['hba1c'] > 6.5:        score += 3
    if row['bmi'] > 30:           score += 2
    if row['age'] > 45:           score += 1
    if row['diabetes_history']:   score += 2
    if row['physical_activity'] == 0: score += 1
    return 1 if score >= 4 else 0

# 2. Cardiovascular Disease Risk
def cvd_risk(row):
    score = 0
    if row['cholesterol'] > 240:    score += 2
    if row['blood_pressure'] > 140: score += 2
    if row['heart_rate'] > 100:     score += 1
    if row['smoker']:               score += 2
    if row['age'] > 55:             score += 2
    if row['bmi'] > 28:             score += 1
    if row['family_history']:       score += 1
    return 1 if score >= 4 else 0

# 3. Kidney Disease (CKD) Risk
def ckd_risk(row):
    score = 0
    if row['creatinine'] > 2.0:     score += 3
    if row['blood_pressure'] > 130: score += 2
    if row['blood_sugar'] > 180:    score += 2
    if row['age'] > 50:             score += 1
    if row['diabetes_history']:     score += 1
    return 1 if score >= 4 else 0

# 4. Hypertension Risk
def hypertension_risk(row):
    score = 0
    if row['blood_pressure'] > 130: score += 3
    if row['bmi'] > 28:             score += 2
    if row['age'] > 40:             score += 1
    if row['smoker']:               score += 1
    if row['alcohol']:              score += 1
    if row['physical_activity']==0: score += 1
    return 1 if score >= 4 else 0

# 5. Fatty Liver Risk
def fatty_liver_risk(row):
    score = 0
    if row['bmi'] > 30:            score += 3
    if row['triglycerides'] > 200: score += 2
    if row['alt_enzyme'] > 80:     score += 2
    if row['alcohol']:             score += 2
    if row['blood_sugar'] > 150:   score += 1
    return 1 if score >= 4 else 0

df['diabetes']      = df.apply(diabetes_risk, axis=1)
df['cvd']           = df.apply(cvd_risk, axis=1)
df['ckd']           = df.apply(ckd_risk, axis=1)
df['hypertension']  = df.apply(hypertension_risk, axis=1)
df['fatty_liver']   = df.apply(fatty_liver_risk, axis=1)

df.to_csv('data/multidisease_data.csv', index=False)

print("✅ Multi-Disease Dataset Created!")
print(f"Total patients: {len(df)}")
print("\nDisease Prevalence:")
for d in ['diabetes','cvd','ckd','hypertension','fatty_liver']:
    count = df[d].sum()
    print(f"  {d:15s}: {count} patients ({count/n*100:.1f}%)")
print("\nSample:")
print(df.head(3))

