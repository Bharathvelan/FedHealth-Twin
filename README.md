
# 🏥 FedHealth-Twin

## AI-Based Smart Health Risk Prediction System

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Project Overview

FedHealth-Twin is an advanced AI/ML system that predicts health risk using
multimodal data while preserving patient privacy through Federated Learning.

---

## ✨ Key Features

- 🔒 **Privacy First** — Federated Learning (data never leaves local system)
- 🧠 **Multimodal AI** — Tabular + Time-Series + Text data fusion
- 📊 **Explainable AI** — SHAP analysis for every prediction
- 💡 **Counterfactual** — Suggests what changes reduce risk
- 📄 **Auto Report** — Generates human-readable health report
- 🌐 **Web Dashboard** — Interactive Streamlit interface

---

## 🏗️ System Architecture

```
FedHealth-Twin/
│
├── 📁 data/
│   ├── health_data.csv
│   ├── doctor_notes.csv
│   └── (other processed files)
│
├── 📁 models/
│   ├── xgboost_model.pkl
│   ├── lstm_model.pth
│   ├── fusion_model.pth
│   ├── federated_model.pth
│   ├── tfidf_vectorizer.pkl
│   ├── svd_reducer.pkl
│   └── scaler.pkl
│
├── 📁 reports/
│   ├── shap_global.png
│   ├── shap_patient.png
│   ├── counterfactual.txt
│   └── health_report.txt
│
├── 🐍 generate_data.py
├── 🐍 tabular_model.py
├── 🐍 timeseries_model.py
├── 🐍 text_encoder.py
├── 🐍 fusion_model.py
├── 🐍 federated_learning.py
├── 🐍 explainability.py
├── 🐍 report_generator.py
├── 🐍 dashboard.py
├── 📄 requirements.txt
└── 📄 README.md
```

---

## 🛠️ Tech Stack

| Module            | Technology                  |
| ----------------- | --------------------------- |
| Tabular Model     | XGBoost                     |
| Time-Series Model | PyTorch LSTM                |
| Text Encoder      | TF-IDF + SVD                |
| Fusion Layer      | PyTorch Neural Network      |
| Privacy           | Federated Learning (FedAvg) |
| Explainability    | SHAP + Counterfactual       |
| Dashboard         | Streamlit                   |

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Bharathvelan/FedHealth-Twin.git
cd FedHealth-Twin
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train all models

```bash
python generate_data.py
python tabular_model.py
python timeseries_model.py
python text_encoder.py
python fusion_model.py
python federated_learning.py
python explainability.py
```

### 4. Launch dashboard

```bash
streamlit run dashboard.py
```

---

## 👥 Team

**Third year Year Project — AI & ML — 2025-2026**

```


```
