import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─── Load Dataset ─────────────────────────────────────────────
df = pd.read_csv('data/multidisease_data.csv')

# Features
feature_cols = ['age','gender','bmi','blood_pressure','blood_sugar',
                'cholesterol','heart_rate','spo2','creatinine','hba1c',
                'triglycerides','alt_enzyme','smoker','diabetes_history',
                'family_history','alcohol','physical_activity']

# Disease targets
diseases = ['diabetes','cvd','ckd','hypertension','fatty_liver']

X = df[feature_cols]
Y = df[diseases]

# Scale features
scaler_multi = StandardScaler()
X_scaled = scaler_multi.fit_transform(X)

# Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42
)

print("🔄 Training Multi-Disease Models...")
print("="*55)

models = {}
results = {}

for disease in diseases:
    y_train = Y_train[disease]
    y_test  = Y_test[disease]

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    models[disease] = model
    results[disease] = acc

    disease_name = disease.replace('_',' ').title()
    print(f"✅ {disease_name:20s} Accuracy: {acc*100:.2f}%")

print("="*55)
avg_acc = np.mean(list(results.values()))
print(f"📊 Average Accuracy: {avg_acc*100:.2f}%")

# Save all models
with open('models/multidisease_models.pkl','wb') as f:
    pickle.dump(models, f)

with open('models/multidisease_scaler.pkl','wb') as f:
    pickle.dump(scaler_multi, f)

# Save test data
X_test_df = pd.DataFrame(X_test, columns=feature_cols)
X_test_df.to_csv('data/X_test_multi.csv', index=False)
Y_test.to_csv('data/Y_test_multi.csv', index=False)

print("\n✅ All multi-disease models saved!")
print("✅ models/multidisease_models.pkl")
print("✅ models/multidisease_scaler.pkl")
