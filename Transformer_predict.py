# -*- coding: utf-8 -*-
# @Time    : 2025/4/28
# @Author  : Bruam1
# @Email   : grey040612@gmail.com
# @File    : Transformer_predict.py
# @Software: Vscode

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("water_data\\fallraw_63000200.csv")
# 缺失值处理：对所有特征列进行线性插值
feature_cols = ["temperature", "humidity", "windpower", "rains", "waterlevels"]
df[feature_cols] = df[feature_cols].interpolate(method='linear', limit_direction='both')
# 选择需要的特征列
data = df[feature_cols].head(1000).values  # shape: (1000, 5)

# 归一化
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# 构造序列
def create_sequences(data, window_size=30):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, :])  # 多特征
        y.append(data[i, -1])  # 只预测水位
    return np.array(X), np.array(y).reshape(-1, 1)

seq_len = 30
X, y = create_sequences(scaled, seq_len)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# 划分训练和测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 定义位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

# 定义Transformer模型
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, model_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(model_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, model_dim)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        out = x[-1]  # 取最后一个时间步的输出
        out = self.output_proj(out)
        return out

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
input_dim = len(feature_cols)
model = TimeSeriesTransformer(input_dim=input_dim).to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 超参数
epochs = 50
batch_size = 32

# 开始训练
for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train.size(0))
    epoch_loss = 0

    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / (X_train.size(0) // batch_size)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

# 预测
model.eval()
with torch.no_grad():
    test_inputs = X_test.to(device)
    predictions = model(test_inputs).cpu().numpy()
    actual = y_test.numpy()

# 反归一化
# 只反归一化水位（最后一列）
predicted_unscaled = scaler.inverse_transform(
    np.concatenate([np.zeros((predictions.shape[0], scaler.n_features_in_ - 1)), predictions], axis=1)
)[:, -1].reshape(-1, 1)
actual_unscaled = scaler.inverse_transform(
    np.concatenate([np.zeros((actual.shape[0], scaler.n_features_in_ - 1)), actual], axis=1)
)[:, -1].reshape(-1, 1)

# 计算残差（绝对误差）
residuals = np.abs(actual_unscaled - predicted_unscaled)

# 自动确定异常阈值（平均值 + 2倍标准差）
mean_res = np.mean(residuals)
std_res = np.std(residuals)
threshold = mean_res + 2 * std_res
print("异常阈值：", threshold)

# 找到异常点
anomalies = np.where(residuals > threshold)[0]

# 输出异常信息
print("\n检测到的水位异常点：")
for idx in anomalies:
    time_step = idx + len(actual_unscaled)  # 若使用全数据时可加偏移
    print(f"时间步 {idx}: 实际水位={actual_unscaled[idx][0]:.2f}, 预测水位={predicted_unscaled[idx][0]:.2f}, 残差={residuals[idx][0]:.2f}")

# 可视化异常点
plt.scatter(anomalies, actual_unscaled[anomalies], color='red', label='Anomalies', zorder=5)
plt.legend()
plt.show()

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(range(len(actual_unscaled)), actual_unscaled, label="Actual")
plt.plot(range(len(predicted_unscaled)), predicted_unscaled, label="Predicted")
plt.title("Transformer Water Level Prediction")
plt.xlabel("Time Step")
plt.ylabel("Water Level")
plt.legend()
plt.grid(True)
plt.show()