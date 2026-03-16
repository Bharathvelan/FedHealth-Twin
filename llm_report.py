import anthropic
import os

def generate_llm_report(patient_name, age, gender, bmi,
                         blood_pressure, blood_sugar, cholesterol,
                         heart_rate, spo2, creatinine, hba1c,
                         triglycerides, alt_enzyme,
                         smoker, diabetes_history, family_history,
                         alcohol, physical_activity,
                         predictions, probabilities, language='en'):

    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    # Build disease summary
    disease_names = {
        'diabetes':    'Diabetes',
        'cvd':         'Cardiovascular Disease',
        'ckd':         'Kidney Disease (CKD)',
        'hypertension':'Hypertension',
        'fatty_liver': 'Fatty Liver'
    }

    risk_diseases  = [disease_names[d] for d,p in predictions.items() if p==1]
    safe_diseases  = [disease_names[d] for d,p in predictions.items() if p==0]
    risk_str       = ', '.join(risk_diseases) if risk_diseases else 'None'
    safe_str       = ', '.join(safe_diseases) if safe_diseases else 'None'

    # Language instruction
    lang_instruction = {
        'en': 'Write the report in clear, simple English.',
        'ta': 'Write the report in Tamil language (தமிழ்). Use simple Tamil words that patients can understand.',
        'hi': 'Write the report in Hindi language (हिंदी). Use simple Hindi words that patients can understand.'
    }

    prompt = f"""
You are a compassionate AI medical assistant helping patients understand their health risk report.

PATIENT DATA:
- Name: {patient_name}
- Age: {age} years, Gender: {gender}
- BMI: {bmi}
- Blood Pressure: {blood_pressure} mmHg
- Blood Sugar: {blood_sugar} mg/dL
- HbA1c: {hba1c}%
- Cholesterol: {cholesterol} mg/dL
- Triglycerides: {triglycerides} mg/dL
- ALT Enzyme: {alt_enzyme} U/L
- Heart Rate: {heart_rate} bpm
- SpO2: {spo2}%
- Creatinine: {creatinine} mg/dL
- Smoker: {'Yes' if smoker else 'No'}
- Diabetes History: {'Yes' if diabetes_history else 'No'}
- Family History: {'Yes' if family_history else 'No'}
- Alcohol: {'Yes' if alcohol else 'No'}
- Physical Activity: {physical_activity}

AI PREDICTION RESULTS:
- Diseases AT RISK: {risk_str}
- Diseases SAFE: {safe_str}
- Risk Probabilities: {', '.join([f"{disease_names[d]}: {p:.1f}%" for d,p in probabilities.items()])}

{lang_instruction.get(language, lang_instruction['en'])}

Please generate a warm, friendly, easy-to-understand health report that includes:

1. A personal greeting to {patient_name}
2. A simple summary of their overall health status
3. Explanation of each disease risk in simple non-medical language
4. What caused each risk (which specific values are concerning)
5. Personalized lifestyle recommendations for each at-risk disease
6. Motivational message to encourage the patient
7. Clear disclaimer that this is AI-generated

Use simple language a non-medical person can understand.
Avoid complex medical jargon.
Be empathetic, supportive and encouraging.
Format with clear sections and bullet points.
Keep it under 600 words.
"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text

    except Exception as e:
        return f"❌ Error generating LLM report: {str(e)}\n\nPlease check your API key."


def generate_llm_summary(predictions, probabilities, patient_name):
    """Generate a short 2-3 sentence AI summary for the dashboard"""
    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    disease_names = {
        'diabetes':    'Diabetes',
        'cvd':         'Cardiovascular Disease',
        'ckd':         'Kidney Disease',
        'hypertension':'Hypertension',
        'fatty_liver': 'Fatty Liver'
    }

    risk_diseases = [disease_names[d] for d,p in predictions.items() if p==1]
    risk_str      = ', '.join(risk_diseases) if risk_diseases else 'no diseases'

    prompt = f"""
Patient {patient_name} has been analysed for 5 diseases.
At risk: {risk_str}
Probabilities: {', '.join([f"{disease_names[d]}: {p:.1f}%" for d,p in probabilities.items()])}

Write ONLY 2-3 short, clear, encouraging sentences summarizing this patient's health status.
Be warm and supportive. No medical jargon. No bullet points. Just plain sentences.
"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            messages=[{"role":"user","content":prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"AI summary unavailable: {str(e)}"


# ─── Test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    test_predictions = {
        'diabetes':    1,
        'cvd':         1,
        'ckd':         0,
        'hypertension':1,
        'fatty_liver': 0
    }
    test_probabilities = {
        'diabetes':    78.5,
        'cvd':         65.2,
        'ckd':         22.1,
        'hypertension':81.3,
        'fatty_liver': 18.4
    }

    print("🔄 Generating LLM Health Report...")
    print("="*60)

    report = generate_llm_report(
        patient_name="Bharath",
        age=45, gender="Male",
        bmi=32.5, blood_pressure=155,
        blood_sugar=280, cholesterol=260,
        heart_rate=95, spo2=93.5,
        creatinine=1.8, hba1c=8.2,
        triglycerides=280, alt_enzyme=65,
        smoker=True, diabetes_history=True,
        family_history=True, alcohol=False,
        physical_activity="Sedentary",
        predictions=test_predictions,
        probabilities=test_probabilities,
        language='en'
    )

    print(report)
    print("="*60)
    print("\n🔄 Generating Short Summary...")
    summary = generate_llm_summary(
        test_predictions, test_probabilities, "Bharath"
    )
    print(summary)
    print("\n✅ LLM Report Generator working!")
