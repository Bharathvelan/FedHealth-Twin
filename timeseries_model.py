import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pickle

# Generate synthetic time-series data (heart rate + SpO2 over 24 timesteps)
np.random.seed(42)
n_samples = 1000
timesteps = 24
n_features = 2  # heart rate, spo2

def generate_timeseries(n, timesteps):
    X = []
    y = []
    for i in range(n):
        risk = np.random.choice([0, 1, 2])
        if risk == 0:  # Low risk - normal vitals
            hr = np.random.normal(75, 5, timesteps)
            spo2 = np.random.normal(98, 1, timesteps)
        elif risk == 1:  # Medium risk
            hr = np.random.normal(90, 8, timesteps)
            spo2 = np.random.normal(95, 2, timesteps)
        else:  # High risk
            hr = np.random.normal(110, 10, timesteps)
            spo2 = np.random.normal(91, 3, timesteps)
        
        sequence = np.column_stack([hr, spo2])
        X.append(sequence)
        y.append(risk)
    
    return np.array(X), np.array(y)

X, y = generate_timeseries(n_samples, timesteps)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# PyTorch Dataset
class VitalSignsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = VitalSignsDataset(X_train, y_train)
test_dataset = VitalSignsDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# LSTM Model
class LSTMHealthModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, num_classes=3):
        super(LSTMHealthModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=0.3)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Last timestep
        out = self.batch_norm(out)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Training
model = LSTMHealthModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

print("🔄 Training LSTM Model...")
epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    scheduler.step()
    
    if (epoch + 1) % 5 == 0:
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                output = model(X_batch)
                _, predicted = torch.max(output, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        acc = correct / total * 100
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f} | Test Accuracy: {acc:.2f}%")

# Save model
torch.save(model.state_dict(), 'models/lstm_model.pth')

# Save test data
np.save('data/X_test_ts.npy', X_test)
np.save('data/y_test_ts.npy', y_test)

print("✅ LSTM Model saved to models/lstm_model.pth")

