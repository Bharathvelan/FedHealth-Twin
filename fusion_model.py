import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from timeseries_model import LSTMHealthModel

# ─── Load All Data ───────────────────────────────────────────
# Tabular data
df = pd.read_csv('data/health_data.csv')
X_tabular = df.drop('risk_label', axis=1).values
y_tabular = df['risk_label'].values

# Time-series data
X_ts = np.load('data/X_test_ts.npy')
y_ts = np.load('data/y_test_ts.npy')

# Text features
X_text = np.load('data/text_features.npy')
y_text = np.load('data/text_labels.npy')

# ─── Get Tabular Features from XGBoost ───────────────────────
with open('models/xgboost_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# Get probability scores from XGBoost (3 classes)
xgb_probs = xgb_model.predict_proba(X_tabular)  # shape: (1000, 3)
print(f"✅ XGBoost features: {xgb_probs.shape}")

# ─── Get Time-Series Features from LSTM ──────────────────────
lstm_model = LSTMHealthModel()
lstm_model.load_state_dict(torch.load('models/lstm_model.pth'))
lstm_model.eval()

# We need full 1000 samples for fusion - regenerate
from timeseries_model import generate_timeseries
X_ts_full, y_ts_full = generate_timeseries(1000, 24)

X_ts_tensor = torch.FloatTensor(X_ts_full)
with torch.no_grad():
    lstm_out = lstm_model(X_ts_tensor)
    lstm_probs = torch.softmax(lstm_out, dim=1).numpy()  # shape: (1000, 3)

print(f"✅ LSTM features: {lstm_probs.shape}")

# ─── Text Features ────────────────────────────────────────────
print(f"✅ Text features: {X_text.shape}")

# ─── Fusion: Concatenate All Features ────────────────────────
# XGBoost probs (3) + LSTM probs (3) + Text features (32) = 38 features
X_fused = np.concatenate([xgb_probs, lstm_probs, X_text], axis=1)
print(f"✅ Fused features shape: {X_fused.shape}")

# Use tabular labels as ground truth
y = y_tabular

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_fused, y, test_size=0.2, random_state=42, stratify=y
)

# ─── PyTorch Fusion Neural Network ───────────────────────────
class FusionNet(nn.Module):
    def __init__(self, input_dim=38, num_classes=3):
        super(FusionNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

# Convert to tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t  = torch.FloatTensor(X_test)
y_test_t  = torch.LongTensor(y_test)

# DataLoader
from torch.utils.data import TensorDataset, DataLoader
train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# Train Fusion Model
fusion_model = FusionNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(fusion_model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

print("\n🔄 Training Fusion Model...")
epochs = 40
for epoch in range(epochs):
    fusion_model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = fusion_model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    if (epoch + 1) % 10 == 0:
        fusion_model.eval()
        with torch.no_grad():
            out = fusion_model(X_test_t)
            _, predicted = torch.max(out, 1)
            acc = accuracy_score(y_test_t.numpy(), predicted.numpy())
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f} | Accuracy: {acc*100:.2f}%")

# ─── Final Evaluation ─────────────────────────────────────────
fusion_model.eval()
with torch.no_grad():
    out = fusion_model(X_test_t)
    _, predicted = torch.max(out, 1)

print("\n✅ Final Fusion Model Results:")
print(classification_report(y_test_t.numpy(), predicted.numpy(),
      target_names=['Low Risk', 'Medium Risk', 'High Risk']))

# Save Fusion Model
torch.save(fusion_model.state_dict(), 'models/fusion_model.pth')

# Save fused data for SHAP
np.save('data/X_fused_train.npy', X_train)
np.save('data/X_fused_test.npy', X_test)
np.save('data/y_fused_test.npy', y_test)

print("✅ Fusion model saved to models/fusion_model.pth")

