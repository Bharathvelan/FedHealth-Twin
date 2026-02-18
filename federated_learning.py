import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import copy

# ─── Simple Federated Neural Network ─────────────────────────
class FederatedNet(nn.Module):
    def __init__(self, input_dim=12, num_classes=3):
        super(FederatedNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

# ─── Load and Split Data into Hospital Clients ────────────────
df = pd.read_csv('data/health_data.csv')
X = df.drop('risk_label', axis=1).values.astype(np.float32)
y = df['risk_label'].values.astype(np.int64)

# Normalize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save scaler
import pickle
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Split into 3 hospital clients
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Divide training data among 3 hospitals
n = len(X_train)
split = n // 3

hospital_data = {
    'Hospital_A': (X_train[:split], y_train[:split]),
    'Hospital_B': (X_train[split:2*split], y_train[split:2*split]),
    'Hospital_C': (X_train[2*split:], y_train[2*split:]),
}

print("📊 Data Distribution across Hospitals:")
for name, (X_h, y_h) in hospital_data.items():
    print(f"   {name}: {len(X_h)} patients")

# ─── Local Training Function ──────────────────────────────────
def local_train(model, X_local, y_local, epochs=5, lr=0.001):
    model = copy.deepcopy(model)
    model.train()
    
    X_t = torch.FloatTensor(X_local)
    y_t = torch.LongTensor(y_local)
    
    dataset = TensorDataset(X_t, y_t)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss   = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
    
    return model.state_dict()

# ─── FedAvg Aggregation ───────────────────────────────────────
def federated_average(global_model, local_weights):
    avg_weights = copy.deepcopy(local_weights[0])
    
    for key in avg_weights.keys():
        for i in range(1, len(local_weights)):
            avg_weights[key] += local_weights[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(local_weights))
    
    global_model.load_state_dict(avg_weights)
    return global_model

# ─── Evaluate Function ────────────────────────────────────────
def evaluate(model, X_test, y_test):
    model.eval()
    X_t = torch.FloatTensor(X_test)
    with torch.no_grad():
        output = model(X_t)
        _, predicted = torch.max(output, 1)
    return accuracy_score(y_test, predicted.numpy())

# ─── Federated Learning Loop ──────────────────────────────────
print("\n🔄 Starting Federated Learning...")
global_model = FederatedNet(input_dim=12, num_classes=3)

rounds = 10
for round_num in range(1, rounds + 1):
    local_weights = []
    
    # Each hospital trains locally
    for hospital_name, (X_h, y_h) in hospital_data.items():
        local_w = local_train(global_model, X_h, y_h, epochs=5)
        local_weights.append(local_w)
    
    # Aggregate weights using FedAvg
    global_model = federated_average(global_model, local_weights)
    
    # Evaluate global model
    acc = evaluate(global_model, X_test, y_test)
    print(f"   Round [{round_num}/{rounds}] Global Model Accuracy: {acc*100:.2f}%")

# ─── Save Federated Model ─────────────────────────────────────
torch.save(global_model.state_dict(), 'models/federated_model.pth')
print("\n✅ Federated Learning complete!")
print("✅ Federated model saved to models/federated_model.pth")

# Final accuracy
final_acc = evaluate(global_model, X_test, y_test)
print(f"✅ Final Global Model Accuracy: {final_acc*100:.2f}%")
