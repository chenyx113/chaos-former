import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 生成Lorenz数据（复用之前的代码）
def generate_lorenz_data():
    sigma = 10.0
    rho = 28.0
    beta = 8.0/3
    y0 = [1.0, 1.0, 1.0]
    h = 0.01
    t_end = 50

    def lorenz_deriv(t, xyz):
        x, y, z = xyz
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z
        return np.array([dx_dt, dy_dt, dz_dt])

    def rk4_integrate(func, y0, t0, t_end, h):
        steps = int(np.ceil((t_end - t0) / h))
        t = np.zeros(steps + 1)
        y = np.zeros((steps + 1, len(y0)))
        t[0] = t0
        y[0] = y0
        for i in range(steps):
            k1 = func(t[i], y[i])
            k2 = func(t[i], y[i] + k1 * h/2)
            k3 = func(t[i], y[i] + k2 * h/2)
            k4 = func(t[i] + h, y[i] + k3 * h)
            y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) * h / 6
            t[i+1] = t[i] + h
        return t, y

    t, y = rk4_integrate(lorenz_deriv, y0, 0, t_end, h)
    return y[:, 0], y[:, 1], y[:, 2]

# 数据预处理
def prepare_data(x, y, z, window_size=10):
    scaler = MinMaxScaler()
    xy = np.column_stack((x, y))
    xy_normalized = scaler.fit_transform(xy)
    z_normalized = scaler.fit_transform(z.reshape(-1, 1))

    X = []
    Z = []
    for i in range(window_size, len(xy_normalized)):
        X.append(xy_normalized[i-window_size:i])
        Z.append(z_normalized[i])

    return np.array(X), np.array(Z), scaler

# Transformer模型定义
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, src):
        src = self.input_proj(src)
        output = self.transformer_encoder(src)
        output = self.output_proj(output[:, -1, :])  # 取最后一个时间步
        return output

# 训练参数设置
window_size = 20
batch_size = 64
epochs = 100
learning_rate = 1e-4

# 生成并预处理数据
x, y, z = generate_lorenz_data()
X, Z, scaler = prepare_data(x, y, z, window_size)

# 划分训练集和测试集
X_train, X_test, Z_train, Z_test = train_test_split(
    X, Z, test_size=0.2, shuffle=False)

# 转换为PyTorch张量
train_data = torch.utils.data.TensorDataset(
    torch.FloatTensor(X_train), torch.FloatTensor(Z_train))
test_data = torch.utils.data.TensorDataset(
    torch.FloatTensor(X_test), torch.FloatTensor(Z_test))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

# 初始化模型、损失函数和优化器
model = TransformerModel(input_dim=2, output_dim=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
train_losses = []
test_losses = []
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_Z in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Z)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # 验证
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_X, batch_Z in test_loader:
            outputs = model(batch_X)
            test_loss += criterion(outputs, batch_Z).item()

    train_losses.append(train_loss/len(train_loader))
    test_losses.append(test_loss/len(test_loader))
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.4f} | Test Loss: {test_losses[-1]:.4f}")

# 可视化训练过程
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# 预测并反归一化
with torch.no_grad():
    test_pred = model(torch.FloatTensor(X_test))

z_pred = scaler.inverse_transform(test_pred.numpy())
z_true = scaler.inverse_transform(Z_test)

# 可视化预测结果
plt.figure(figsize=(12, 6))
plt.plot(z_true[:500], label='True Values', alpha=0.7)
plt.plot(z_pred[:500], label='Predictions', alpha=0.7)
plt.xlabel('Time Step')
plt.ylabel('z Value')
plt.title('True vs Predicted z Values')
plt.legend()
plt.show()

# 误差分析
errors = z_true - z_pred

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(z_true, z_pred, s=5, alpha=0.5)
plt.plot([z_true.min(), z_true.max()], [z_true.min(), z_true.max()], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Prediction vs True Value')

plt.subplot(1, 2, 2)
plt.hist(errors, bins=50)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.tight_layout()
plt.show()

# 统计指标
mse = np.mean(errors**2)
mae = np.mean(np.abs(errors))
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Max Absolute Error: {np.max(np.abs(errors)):.4f}")
