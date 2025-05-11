# -*- coding: utf-8 -*-
# @Time    : 2025/4/28 10:34
# @Author  : Bruam1
# @Email   : grey040612@gmail.com
# @File    : LSTM_predict.py
# @Software: Vscode


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("water_data\\fallraw_63000200.csv")
data = df["waterlevels"].head(1000).values.reshape(-1, 1)

# print(data)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def create_dataset(dataset, look_back = 10):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i: i + look_back])
        y.append(dataset[i + look_back])
    return np.array(X), np.array(y)

look_back = 10
X, y = create_dataset(data_scaled, look_back)

X = torch.tensor(X, dtype = torch.float32)
y = torch.tensor(y, dtype = torch.float32)

X = X.view(X.shape[0], X.shape[1], 1)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

class LSTMNet(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 50, output_size = 1):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first = True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
    
model = LSTMNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

epochs = 10 
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1} / {epochs}, Loss: {loss.item():.6f}")

model.eval()
with torch.no_grad():
    y_pred = model(X_test)

y_pred_np = scaler.inverse_transform(y_pred.numpy())
y_test_np = scaler.inverse_transform(y_test.numpy())

plt.figure(figsize = (10, 6))
plt.plot(y_test_np, label = "Actual")
plt.plot(y_pred_np, label = "Predicted")
plt.title("Water Level Prediction (LSTM - Pytorch)")
plt.legend()
plt.grid()
plt.show()