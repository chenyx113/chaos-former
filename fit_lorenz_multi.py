import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# 生成Lorenz数据（保持与之前相同）
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

# 改进的多模态Scaler
class MultiModalScaler:
    def __init__(self):
        self.scalers = [StandardScaler() for _ in range(3)]

    def fit_transform(self, data):
        scaled = np.zeros_like(data)
        for i in range(3):
            scaled[:, i] = self.scalers[i].fit_transform(data[:, i].reshape(-1, 1)).flatten()
        return scaled

    def inverse_transform(self, data, modality):
        return self.scalers[modality].inverse_transform(data.reshape(-1, 1))

# 修复后的数据集类
class LorenzDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size=30, mask_ratio=0.3):
        self.data = data
        self.window_size = window_size
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        window = self.data[idx:idx+self.window_size]

        # 生成固定大小的掩码矩阵
        time_mask = np.random.rand(self.window_size, 3) < self.mask_ratio
        masked_window = window.copy()
        masked_window[time_mask] = 0

        # 转换为张量
        return (
            torch.FloatTensor(masked_window),  # 掩码后的输入
            torch.FloatTensor(window),         # 原始数据
            torch.BoolTensor(time_mask)        # 掩码位置标识
        )

# 自定义collate_fn处理变长目标
def collate_fn(batch):
    inputs = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    masks = torch.stack([item[2] for item in batch])
    return inputs, targets, masks

# 改进的Transformer模型
class MultiModalTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(3, d_model)

        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=512, dropout=0.1
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, src, src_mask=None):
        # src: (batch, seq_len, 3)
        src = self.input_proj(src)
        src = src.permute(1, 0, 2)  # (seq_len, batch, d_model)

        memory = self.transformer(src, src_key_padding_mask=src_mask)
        memory = memory.permute(1, 0, 2)  # (batch, seq_len, d_model)

        return self.output_proj(memory)  # (batch, seq_len, 3)

# 训练参数
window_size = 30
batch_size = 64
epochs = 100
learning_rate = 1e-4

# 数据准备
x, y, z = generate_lorenz_data()
data = np.column_stack([x, y, z])

scaler = MultiModalScaler()
scaled_data = scaler.fit_transform(data)

dataset = LorenzDataset(scaled_data, window_size=window_size)
train_size = int(0.8 * len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, len(dataset) - train_size])

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# 模型初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiModalTransformer().to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# 修复后的训练循环
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for inputs, targets, masks in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        # 只计算被掩码位置的损失
        loss = torch.mean((outputs[masks] - targets[masks])**2)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# 验证循环
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets, masks in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            outputs = model(inputs)
            loss = torch.mean((outputs[masks] - targets[masks])**2)
            total_loss += loss.item()

    return total_loss / len(loader)

# 训练过程（保持不变）
train_losses = []
test_losses = []
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer)
    test_loss = evaluate(model, test_loader)

    scheduler.step()

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Test: {test_loss:.4f}")

# 可视化训练过程（保持不变）
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training Progress')
plt.show()

def predict(model, input_data, mask_pattern, steps=1):
    model.eval()
    predictions = []

    current_window = torch.FloatTensor(input_data).unsqueeze(0).to(device)

    # 维度检查与调整
    if isinstance(mask_pattern, np.ndarray):
        mask_pattern = torch.from_numpy(mask_pattern)
    if mask_pattern.dim() == 2:
        mask_pattern = mask_pattern.unsqueeze(0)  # 添加批次维度
    mask_pattern = mask_pattern.to(device)

    for _ in range(steps):
        with torch.no_grad():
            outputs = model(current_window)

        # 使用布尔掩码进行索引
        predicted = current_window.clone()
        update_mask = mask_pattern[:, :current_window.size(1)]  # 适应可能变化的窗口长度
        predicted[update_mask] = outputs[update_mask]

        # 窗口滑动更新
        current_window = torch.cat([
            current_window[:, 1:],
            predicted[:, -1:]
        ], dim=1)

        predictions.append(predicted[0, -1].cpu().numpy())

    return np.array(predictions)

window_size = 30
# 使用示例：用x,y预测z
test_sample = scaled_data[-window_size-10:-10]
mask_pattern = np.zeros((1, window_size, 3), dtype=bool)
mask_pattern[0, :, 2] = True  # 掩盖整个窗口的z值

pred_scaled = predict(model, test_sample, mask_pattern, steps=10)

pred_z = scaler.inverse_transform(pred_scaled[:, 2], 2)
true_z = scaler.inverse_transform(scaled_data[-10:, 2], 2)

# 绘制结果
plt.figure(figsize=(12,6))
plt.plot(true_z, label='True z')
plt.plot(pred_z, label='Predicted z')
plt.title('Z Value Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Z Value')
plt.legend()
plt.show()
