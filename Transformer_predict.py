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

plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文和负号
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv("water_data\\fallraw_63000200.csv")
# 缺失值处理：对所有特征列进行线性插值
feature_cols = ["temperature", "humidity", "windpower", "rains", "waterlevels"]
df[feature_cols] = df[feature_cols].interpolate(method='linear', limit_direction='both')
# 选择需要的特征列
data = df[feature_cols].head(101).values  # 用前100条预测第101条

# 归一化
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# 构造单一序列样本
X = scaled[:100, :]  # 前100条作为输入
y = scaled[100, -1]  # 第101条的水位作为目标

X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # shape: (1, 100, 5)
y = torch.tensor([[y]], dtype=torch.float32)           # shape: (1, 1)

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
    def __init__(self, input_dim=1, model_dim=64, num_heads=4, num_layers=2, dropout=0.05):
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
epochs = 20

# 开始训练
for epoch in range(epochs):
    model.train()
    batch_x, batch_y = X.to(device), y.to(device)
    optimizer.zero_grad()
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

# 预测
model.eval()
with torch.no_grad():
    prediction = model(X.to(device)).cpu().numpy()  # shape: (1, 1)
    actual = y.numpy()                             # shape: (1, 1)

# 反归一化
predicted_unscaled = scaler.inverse_transform(
    np.concatenate([np.zeros((1, scaler.n_features_in_ - 1)), prediction], axis=1)
)[:, -1].reshape(-1, 1)
actual_unscaled = scaler.inverse_transform(
    np.concatenate([np.zeros((1, scaler.n_features_in_ - 1)), actual], axis=1)
)[:, -1].reshape(-1, 1)

# 输出预测结果
print(f"\n用前100条预测第101条：\n实际水位={actual_unscaled[0][0]:.2f}, 预测水位={predicted_unscaled[0][0]:.2f}, 残差={abs(actual_unscaled[0][0] - predicted_unscaled[0][0]):.2f}")

# 可视化
plt.figure(figsize=(8, 4))
plt.plot(range(1, 102), scaler.inverse_transform(scaled)[:, -1], label="历史水位")
plt.scatter([101], actual_unscaled, color='green', label='实际第101条', zorder=5)
plt.scatter([101], predicted_unscaled, color='red', label='预测第101条', zorder=5)
plt.legend()
plt.title("用前100条预测第101条水位")
plt.xlabel("时间步")
plt.ylabel("水位")
plt.grid(True)
plt.show()