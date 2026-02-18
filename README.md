# 🏥 FedHealth-Twin

## AI-Based Smart Health Risk Prediction System

### Project Overview

FedHealth-Twin is an advanced AI/ML system that predicts
health risk using multimodal data while preserving patient
privacy through Federated Learning.

### Key Features

- Multi-modal data fusion (Tabular + Time-Series + Text)
- Federated Learning across 3 hospital nodes
- Explainable AI using SHAP analysis
- Counterfactual suggestions for risk reduction
- Automated health report generation
- Interactive Streamlit dashboard

### Tech Stack

- Python 3.13
- PyTorch (LSTM Model)
- XGBoost (Tabular Model)
- Federated Learning (FedAvg)
- SHAP (Explainability)
- Streamlit (Dashboard)

### How to Run

1. Install dependencies:
   pip install -r requirements.txt
2. Generate dataset:
   python generate_data.py
3. Train models:
   python tabular_model.py
   python timeseries_model.py
   python text_encoder.py
   python fusion_model.py
   python federated_learning.py
4. Generate explanations:
   python explainability.py
5. Launch dashboard:
   streamlit run dashboard.py

### Project Structure

FedHealth-Twin/
├── data/               # Datasets
├── models/             # Saved models
├── reports/            # Generated reports
├── generate_data.py    # Data generation
├── tabular_model.py    # XGBoost model
├── timeseries_model.py # PyTorch LSTM
├── text_encoder.py     # Text features
├── fusion_model.py     # Feature fusion
├── federated_learning.py # FL simulation
├── explainability.py   # SHAP analysis
├── report_generator.py # Report generation
└── dashboard.py        # Streamlit UI

### Team

Final Year Project - AI & ML
2024-2025

```

```
