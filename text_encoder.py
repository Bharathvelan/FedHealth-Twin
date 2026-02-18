import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle

# Generate synthetic doctor notes
np.random.seed(42)

low_risk_notes = [
    "Patient is healthy. Normal blood pressure and sugar levels. No complaints.",
    "Routine checkup. All vitals normal. Patient exercises regularly.",
    "No significant findings. Patient maintaining healthy lifestyle.",
    "Blood reports normal. Patient advised to continue current diet.",
    "Regular followup. No symptoms reported. Vitals stable.",
]

medium_risk_notes = [
    "Patient reports occasional chest discomfort. Mild hypertension observed.",
    "Slightly elevated blood sugar. Patient advised diet control and exercise.",
    "Mild obesity noted. Blood pressure borderline high. Monitor regularly.",
    "Patient has family history of diabetes. Sugar levels slightly elevated.",
    "Occasional breathlessness reported. SpO2 borderline. Follow up needed.",
]

high_risk_notes = [
    "Patient has severe hypertension and uncontrolled diabetes. Immediate attention required.",
    "Critical SpO2 levels observed. Patient reports chest pain and fatigue.",
    "High creatinine levels indicating kidney stress. Urgent consultation needed.",
    "Patient has history of cardiac issues. Symptoms worsening. Hospitalization advised.",
    "Severe obesity with uncontrolled blood sugar. High risk of cardiovascular event.",
]

# Generate 1000 notes with labels
notes = []
labels = []

for i in range(1000):
    risk = np.random.choice([0, 1, 2])
    if risk == 0:
        note = np.random.choice(low_risk_notes)
    elif risk == 1:
        note = np.random.choice(medium_risk_notes)
    else:
        note = np.random.choice(high_risk_notes)
    
    # Add some variation
    note = note + f" Patient ID: {i}."
    notes.append(note)
    labels.append(risk)

# Save notes and labels
df_notes = pd.DataFrame({'note': notes, 'label': labels})
df_notes.to_csv('data/doctor_notes.csv', index=False)

# TF-IDF Vectorizer
print("🔄 Training Text Encoder...")
vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
X_tfidf = vectorizer.fit_transform(notes)

# Reduce dimensions to 32 features using SVD
svd = TruncatedSVD(n_components=32, random_state=42)
X_text_features = svd.fit_transform(X_tfidf)

print(f"✅ Text features shape: {X_text_features.shape}")

# Save vectorizer and SVD
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('models/svd_reducer.pkl', 'wb') as f:
    pickle.dump(svd, f)

# Save text features
np.save('data/text_features.npy', X_text_features)
np.save('data/text_labels.npy', np.array(labels))

print("✅ Text encoder saved!")
print(f"✅ Doctor notes saved to data/doctor_notes.csv")
print(f"Sample note: {notes[0]}")

