# Multilingual Health Report Translations
# Tamil (ta-IN) and Hindi (hi-IN)

def get_voice_report(patient_name, age, gender, risk_name,
                     confidence, probabilities, predictions,
                     recs, language='en'):

    disease_names = {
        'en': {
            'diabetes':    'Diabetes',
            'cvd':         'Cardiovascular Disease',
            'ckd':         'Kidney Disease',
            'hypertension':'Hypertension',
            'fatty_liver': 'Fatty Liver'
        },
        'ta': {
            'diabetes':    'நீரிழிவு நோய்',
            'cvd':         'இதய நோய்',
            'ckd':         'சிறுநீரக நோய்',
            'hypertension':'உயர் இரத்த அழுத்தம்',
            'fatty_liver': 'கொழுப்பு கல்லீரல்'
        },
        'hi': {
            'diabetes':    'मधुमेह',
            'cvd':         'हृदय रोग',
            'ckd':         'किडनी रोग',
            'hypertension':'उच्च रक्तचाप',
            'fatty_liver': 'फैटी लीवर'
        }
    }

    risk_translation = {
        'en': {
            'LOW RISK':    'Low Risk',
            'MEDIUM RISK': 'Medium Risk',
            'HIGH RISK':   'High Risk'
        },
        'ta': {
            'LOW RISK':    'குறைந்த ஆபத்து',
            'MEDIUM RISK': 'நடுத்தர ஆபத்து',
            'HIGH RISK':   'அதிக ஆபத்து'
        },
        'hi': {
            'LOW RISK':    'कम जोखिम',
            'MEDIUM RISK': 'मध्यम जोखिम',
            'HIGH RISK':   'उच्च जोखिम'
        }
    }

    if language == 'ta':
        # Tamil Report
        risk_diseases = [disease_names['ta'][d]
                        for d,p in predictions.items() if p == 1]
        risk_str = ', '.join(risk_diseases) if risk_diseases else 'எதுவும் இல்லை'

        report = f"""
வணக்கம் {patient_name}.
நீங்கள் ஃபெட்ஹெல்த் டுவின் AI சுகாதார அமைப்பை பயன்படுத்துகிறீர்கள்.

உங்கள் உடல்நல பரிசோதனை முடிவுகள்:

வயது: {age} ஆண்டுகள்.
பாலினம்: {'ஆண்' if gender == 'Male' else 'பெண்'}.

நோய் பகுப்பாய்வு:
நீரிழிவு நோய் ஆபத்து: {probabilities.get('diabetes', 0):.1f} சதவீதம்.
{'நீரிழிவு நோய் ஆபத்தில் உள்ளீர்கள்.' if predictions.get('diabetes') else 'நீரிழிவு நோயிலிருந்து பாதுகாப்பாக உள்ளீர்கள்.'}

இதய நோய் ஆபத்து: {probabilities.get('cvd', 0):.1f} சதவீதம்.
{'இதய நோய் ஆபத்தில் உள்ளீர்கள்.' if predictions.get('cvd') else 'இதய நோயிலிருந்து பாதுகாப்பாக உள்ளீர்கள்.'}

சிறுநீரக நோய் ஆபத்து: {probabilities.get('ckd', 0):.1f} சதவீதம்.
{'சிறுநீரக நோய் ஆபத்தில் உள்ளீர்கள்.' if predictions.get('ckd') else 'சிறுநீரக நோயிலிருந்து பாதுகாப்பாக உள்ளீர்கள்.'}

உயர் இரத்த அழுத்த ஆபத்து: {probabilities.get('hypertension', 0):.1f} சதவீதம்.
{'உயர் இரத்த அழுத்த ஆபத்தில் உள்ளீர்கள்.' if predictions.get('hypertension') else 'இரத்த அழுத்தம் சீராக உள்ளது.'}

கொழுப்பு கல்லீரல் ஆபத்து: {probabilities.get('fatty_liver', 0):.1f} சதவீதம்.
{'கொழுப்பு கல்லீரல் ஆபத்தில் உள்ளீர்கள்.' if predictions.get('fatty_liver') else 'கல்லீரல் ஆரோக்கியமாக உள்ளது.'}

ஆபத்தில் உள்ள நோய்கள்: {risk_str}.

தயவுசெய்து உங்கள் மருத்துவரை அணுகவும்.
ஆரோக்கியமாக இருங்கள். நன்றி.
        """.strip()

    elif language == 'hi':
        # Hindi Report
        risk_diseases = [disease_names['hi'][d]
                        for d,p in predictions.items() if p == 1]
        risk_str = ', '.join(risk_diseases) if risk_diseases else 'कोई नहीं'

        report = f"""
नमस्ते {patient_name}.
आप फेडहेल्थ ट्विन AI स्वास्थ्य प्रणाली का उपयोग कर रहे हैं।

आपकी स्वास्थ्य जांच के परिणाम:

आयु: {age} वर्ष।
लिंग: {'पुरुष' if gender == 'Male' else 'महिला'}।

रोग विश्लेषण:
मधुमेह जोखिम: {probabilities.get('diabetes', 0):.1f} प्रतिशत।
{'आप मधुमेह के खतरे में हैं।' if predictions.get('diabetes') else 'आप मधुमेह से सुरक्षित हैं।'}

हृदय रोग जोखिम: {probabilities.get('cvd', 0):.1f} प्रतिशत।
{'आप हृदय रोग के खतरे में हैं।' if predictions.get('cvd') else 'आप हृदय रोग से सुरक्षित हैं।'}

किडनी रोग जोखिम: {probabilities.get('ckd', 0):.1f} प्रतिशत।
{'आप किडनी रोग के खतरे में हैं।' if predictions.get('ckd') else 'आप किडनी रोग से सुरक्षित हैं।'}

उच्च रक्तचाप जोखिम: {probabilities.get('hypertension', 0):.1f} प्रतिशत।
{'आप उच्च रक्तचाप के खतरे में हैं।' if predictions.get('hypertension') else 'आपका रक्तचाप सामान्य है।'}

फैटी लीवर जोखिम: {probabilities.get('fatty_liver', 0):.1f} प्रतिशत।
{'आप फैटी लीवर के खतरे में हैं।' if predictions.get('fatty_liver') else 'आपका लीवर स्वस्थ है।'}

जोखिम वाले रोग: {risk_str}।

कृपया अपने डॉक्टर से मिलें।
स्वस्थ रहें। धन्यवाद।
        """.strip()

    else:
        # English Report
        risk_diseases = [disease_names['en'][d]
                        for d,p in predictions.items() if p == 1]
        risk_str = ', '.join(risk_diseases) if risk_diseases else 'None'

        report = f"""
Hello {patient_name}.
This is your FedHealth Twin AI health report.

Patient Details:
Age: {age} years. Gender: {gender}.

Disease Analysis:
Diabetes risk: {probabilities.get('diabetes', 0):.1f} percent.
{'You are at risk for Diabetes.' if predictions.get('diabetes') else 'You are safe from Diabetes.'}

Cardiovascular risk: {probabilities.get('cvd', 0):.1f} percent.
{'You are at risk for Cardiovascular disease.' if predictions.get('cvd') else 'You are safe from Cardiovascular disease.'}

Kidney disease risk: {probabilities.get('ckd', 0):.1f} percent.
{'You are at risk for Kidney disease.' if predictions.get('ckd') else 'You are safe from Kidney disease.'}

Hypertension risk: {probabilities.get('hypertension', 0):.1f} percent.
{'You are at risk for Hypertension.' if predictions.get('hypertension') else 'You are safe from Hypertension.'}

Fatty Liver risk: {probabilities.get('fatty_liver', 0):.1f} percent.
{'You are at risk for Fatty Liver.' if predictions.get('fatty_liver') else 'You are safe from Fatty Liver.'}

Diseases at risk: {risk_str}.
Please consult your doctor immediately.
Stay healthy. Thank you.
        """.strip()

    return report


def get_language_html(en_text, ta_text, hi_text):
    """Returns HTML with language selector + Web Speech API"""

    def escape(t):
        return t.replace("'","\\'").replace('"','\\"').replace('\n',' ').replace('\r','')

    return f"""
    <div style="background:linear-gradient(135deg,#0d1b2a,#1b2838);
         border-radius:18px;padding:22px;
         border:1px solid rgba(255,152,0,0.3);
         box-shadow:0 4px 25px rgba(255,152,0,0.15);
         text-align:center;margin:10px 0;">

        <div style="color:#ffcc02;font-size:1.15rem;font-weight:700;margin-bottom:4px;">
            🎙️ Multilingual AI Voice Assistant
        </div>
        <div style="color:#aaa;font-size:0.8rem;margin-bottom:14px;">
            English | தமிழ் | हिंदी
        </div>

        <!-- Wave Animation -->
        <div id="wave" style="display:none;justify-content:center;
             align-items:center;gap:4px;margin:8px 0;">
            <div style="width:5px;height:12px;border-radius:3px;
                 background:linear-gradient(#ffcc02,#ff9800);
                 animation:wave 1.2s ease-in-out 0.0s infinite;"></div>
            <div style="width:5px;height:24px;border-radius:3px;
                 background:linear-gradient(#ffcc02,#ff9800);
                 animation:wave 1.2s ease-in-out 0.1s infinite;"></div>
            <div style="width:5px;height:18px;border-radius:3px;
                 background:linear-gradient(#ffcc02,#ff9800);
                 animation:wave 1.2s ease-in-out 0.2s infinite;"></div>
            <div style="width:5px;height:30px;border-radius:3px;
                 background:linear-gradient(#ffcc02,#ff9800);
                 animation:wave 1.2s ease-in-out 0.3s infinite;"></div>
            <div style="width:5px;height:22px;border-radius:3px;
                 background:linear-gradient(#ffcc02,#ff9800);
                 animation:wave 1.2s ease-in-out 0.2s infinite;"></div>
            <div style="width:5px;height:16px;border-radius:3px;
                 background:linear-gradient(#ffcc02,#ff9800);
                 animation:wave 1.2s ease-in-out 0.1s infinite;"></div>
            <div style="width:5px;height:10px;border-radius:3px;
                 background:linear-gradient(#ffcc02,#ff9800);
                 animation:wave 1.2s ease-in-out 0.0s infinite;"></div>
        </div>

        <!-- Language Buttons -->
        <div style="display:flex;gap:8px;justify-content:center;
             flex-wrap:wrap;margin:10px 0;">
            <button onclick="speakLang('en')" style="
                background:linear-gradient(135deg,#1565c0,#0d47a1);
                color:white;border:none;border-radius:25px;
                padding:9px 20px;font-size:0.88rem;
                font-weight:600;cursor:pointer;">
                🇬🇧 English
            </button>
            <button onclick="speakLang('ta')" style="
                background:linear-gradient(135deg,#b71c1c,#c62828);
                color:white;border:none;border-radius:25px;
                padding:9px 20px;font-size:0.88rem;
                font-weight:600;cursor:pointer;">
                🇮🇳 Tamil
            </button>
            <button onclick="speakLang('hi')" style="
                background:linear-gradient(135deg,#e65100,#bf360c);
                color:white;border:none;border-radius:25px;
                padding:9px 20px;font-size:0.88rem;
                font-weight:600;cursor:pointer;">
                🇮🇳 Hindi
            </button>
            <button onclick="stopSpeech()" style="
                background:linear-gradient(135deg,#424242,#212121);
                color:white;border:none;border-radius:25px;
                padding:9px 20px;font-size:0.88rem;
                font-weight:600;cursor:pointer;">
                ⏹️ Stop
            </button>
            <button onclick="window.speechSynthesis.pause();
                document.getElementById('wave').style.display='none';" style="
                background:linear-gradient(135deg,#1a237e,#283593);
                color:white;border:none;border-radius:25px;
                padding:9px 20px;font-size:0.88rem;
                font-weight:600;cursor:pointer;">
                ⏸️ Pause
            </button>
            <button onclick="window.speechSynthesis.resume();
                document.getElementById('wave').style.display='flex';" style="
                background:linear-gradient(135deg,#1b5e20,#2e7d32);
                color:white;border:none;border-radius:25px;
                padding:9px 20px;font-size:0.88rem;
                font-weight:600;cursor:pointer;">
                ▶️ Resume
            </button>
        </div>

        <!-- Speed Slider -->
        <div style="display:flex;gap:10px;justify-content:center;
             align-items:center;flex-wrap:wrap;margin-top:8px;">
            <label style="color:#aaa;font-size:0.78rem;">Speed:</label>
            <input type="range" id="rateSlider" min="0.5" max="2"
                   step="0.1" value="0.85" style="width:100px;"
                   oninput="document.getElementById('rv').innerText=
                   parseFloat(this.value).toFixed(1)+'x'"/>
            <span id="rv" style="color:#ffcc02;font-size:0.78rem;">0.9x</span>
        </div>

        <div id="statusMsg"
             style="color:#ffcc02;font-size:0.78rem;margin-top:8px;"></div>
    </div>

    <style>
    @keyframes wave {{
        0%,100%{{ transform:scaleY(0.5); opacity:0.6; }}
        50%    {{ transform:scaleY(1.5); opacity:1.0; }}
    }}
    </style>

    <script>
    var reports = {{
        'en': '{escape(en_text)}',
        'ta': '{escape(ta_text)}',
        'hi': '{escape(hi_text)}'
    }};
    var langCodes = {{
        'en': 'en-US',
        'ta': 'ta-IN',
        'hi': 'hi-IN'
    }};

    function speakLang(lang) {{
        window.speechSynthesis.cancel();
        var text = reports[lang];
        var utt  = new SpeechSynthesisUtterance(text);
        utt.lang = langCodes[lang];
        utt.rate = parseFloat(document.getElementById('rateSlider').value);

        var voices = window.speechSynthesis.getVoices();
        var match  = voices.find(function(v) {{
            return v.lang === langCodes[lang] ||
                   v.lang.startsWith(lang === 'en' ? 'en' :
                                     lang === 'ta' ? 'ta' : 'hi');
        }});
        if (match) utt.voice = match;

        utt.onstart = function() {{
            document.getElementById('wave').style.display = 'flex';
            var labels = {{'en':'English','ta':'Tamil','hi':'Hindi'}};
            document.getElementById('statusMsg').innerText =
                '🎙️ Speaking in ' + labels[lang] + '...';
        }};
        utt.onend = function() {{
            document.getElementById('wave').style.display = 'none';
            document.getElementById('statusMsg').innerText = '✅ Done!';
        }};
        utt.onerror = function(e) {{
            document.getElementById('wave').style.display = 'none';
            document.getElementById('statusMsg').innerText =
                '⚠️ Voice not available for this language on your browser.';
        }};
        window.speechSynthesis.speak(utt);
    }}

    function stopSpeech() {{
        window.speechSynthesis.cancel();
        document.getElementById('wave').style.display = 'none';
        document.getElementById('statusMsg').innerText = '⏹️ Stopped.';
    }}

    window.speechSynthesis.onvoiceschanged = function() {{}};
    </script>
    """
