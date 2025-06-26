# Beginner's Guide: Time Series Forecasting with PatchTST

This guide will walk you through training a modern deep learning model (PatchTST) for time series forecasting, using a workflow similar to the classic LSTM window-split method. We'll cover data loading, processing, training, evaluation, and visualization, all in Python (PyTorch), using the [Time-Series-Library](https://github.com/<your-repo>/Time-Series-Library).

---

## 1. Environment Setup

**On Google Colab:**

```python
!git clone https://github.com/<your-repo>/Time-Series-Library.git
%cd Time-Series-Library
!pip install -r requirements.txt
```

---

## 2. Data Preparation (Window-Split Method)

Suppose you have a CSV file (e.g., `ETTh1.csv`) with a single column of values. The classic window-split method works as follows:
- For each time step, take the previous `k` values as input (window), and the value right after as the label.

**Example:**
- If your series is `[x1, x2, x3, ..., x1000]` and `window_size = 24`, then:
  - Input: `[x1, ..., x24]` → Label: `x25`
  - Input: `[x2, ..., x25]` → Label: `x26`
  - ...

### **Code: Custom Dataset for Window Split**

```python
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class WindowDataset(Dataset):
    def __init__(self, series, window_size):
        self.series = np.array(series)
        self.window_size = window_size

    def __len__(self):
        return len(self.series) - self.window_size

    def __getitem__(self, idx):
        x = self.series[idx:idx+self.window_size]
        y = self.series[idx+self.window_size]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
```

### **Load and Split Data**

```python
# Load your CSV (assume single column 'value')
df = pd.read_csv('dataset/ETT-small/ETTh1.csv')
series = df['value'].values  # Change 'value' to your column name

# Normalize (optional but recommended)
mean, std = series.mean(), series.std()
series = (series - mean) / std

# Split into train/val/test (e.g., 70/15/15)
train_size = int(0.7 * len(series))
val_size = int(0.15 * len(series))
train_series = series[:train_size]
val_series = series[train_size:train_size+val_size]
test_series = series[train_size+val_size:]

window_size = 24  # or any value you like

train_dataset = WindowDataset(train_series, window_size)
val_dataset = WindowDataset(val_series, window_size)
test_dataset = WindowDataset(test_series, window_size)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

---

## 3. Model: PatchTST (Modern Deep Learning)

PatchTST is a transformer-based model for time series. We'll use a simple version for 1D forecasting.

```python
import torch.nn as nn

class SimplePatchTST(nn.Module):
    def __init__(self, input_length):
        super().__init__()
        self.fc1 = nn.Linear(input_length, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x: (batch, window_size)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x.squeeze(-1)

model = SimplePatchTST(window_size)
```

---

## 4. Training Loop

```python
import torch.optim as optim
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 20
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            val_loss += loss.item()
    val_losses.append(val_loss / len(val_loader))
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
```

---

## 5. Visualize Training and Validation Losses

```python
import matplotlib.pyplot as plt

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()
```

---

## 6. Evaluate and Visualize Model Predictions

```python
model.eval()
preds, trues = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        output = model(x)
        preds.append(output.cpu().numpy())
        trues.append(y.numpy())

import numpy as np
preds = np.concatenate(preds)
trues = np.concatenate(trues)

plt.figure(figsize=(15,5))
plt.plot(trues, label='True')
plt.plot(preds, label='Predicted')
plt.legend()
plt.title('Model Forecast vs. Ground Truth')
plt.show()
```

---

## 7. Summary

- **Window-split**: We use a sliding window to create (input, label) pairs, just like in LSTM workflows.
- **Batching**: DataLoader batches samples for efficient training.
- **Model**: PatchTST (or a simple MLP here) can be swapped for any deep learning model.
- **Visualization**: Always plot your losses and predictions to check model performance.

---

**You can now experiment with window sizes, model architectures, and hyperparameters!**

If you want to use the full PatchTST or other advanced models, check the `models/` directory in the repo for ready-to-use implementations. 