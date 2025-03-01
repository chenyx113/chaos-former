#fit_signals.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from dataset_gen import *
from display_signal import *
from models import TransformerModel


# 训练参数设置
window_size = 32
batch_size = 256
epochs = 100
learning_rate = 1e-4
signal_dim = 3

# 生成并预处理数据
x,y,z = generate_signal()

# use x,y to fit z
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
model = TransformerModel(input_dim=signal_dim, output_dim=signal_dim)
mse = nn.MSELoss()
sml1 = nn.SmoothL1Loss(reduction = 'mean')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

######################################################################################

# 训练循环
train_losses = []
test_losses = []
min_loss = 99999
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_Z in train_loader:
        optimizer.zero_grad()
        #idx = 0, mask the x value in this batch training
        #idx = 1, mask the y value in this batch training
        #idx = 2, mask the z value in this batch training
        #idx = np.random.randint(0, signal_dim=3), mask the random dimension value in this batch training
        idx = np.random.randint(0, signal_dim)
        batch_X[:,:,idx] = 0.0
        outputs = model(batch_X)
        #print(batch_X.shape, outputs.shape)
        loss1 = 0.5*mse(outputs, batch_Z)
        loss2 = 0.5*sml1(outputs, batch_Z)
        #print(loss1.item(), loss2.item())
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # 验证
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_X, batch_Z in test_loader:
            outputs = model(batch_X)
            #print(batch_X.shape, outputs.shape)
            loss1 = 0.5*mse(outputs, batch_Z) 
            loss2 = 0.5*sml1(outputs, batch_Z)
            test_loss += (loss1 + loss2).item()
            if test_loss < min_loss:
                min_loss = test_loss
                #torch.save(model, "./checkpoint/best_model_{:.5f}.pt".format(test_loss))
                torch.save(model, "./checkpoint/best_model.pt")

    train_losses.append(train_loss/len(train_loader))
    test_losses.append(test_loss/len(test_loader))
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.5f} | Test Loss: {test_losses[-1]:.5f}")

######################################################################################

# 可视化训练过程
plot_training_process(train_losses, test_losses)
######################################################################################
model_best = torch.load('./checkpoint/best_model.pt')

# 预测并反归一化
with torch.no_grad():
    X_test1 = np.copy(X_test)
    X_test2 = np.copy(X_test)
    X_test3 = np.copy(X_test)

    X_test1[:, :, 2] = 0.0
    X_test2[:, :, 1] = 0.0
    X_test3[:, :, 0] = 0.0

    test_pred1 = model_best(torch.FloatTensor(X_test1))
    test_pred2 = model_best(torch.FloatTensor(X_test2))
    test_pred3 = model_best(torch.FloatTensor(X_test3))

z_pred1 = scaler.inverse_transform(test_pred1.numpy())
z_pred2 = scaler.inverse_transform(test_pred2.numpy())
z_pred3 = scaler.inverse_transform(test_pred3.numpy())
z_true = scaler.inverse_transform(Z_test)

######################################################################################
# 显示test data 真实值和与预测值
plot_truth_vs_prediction(z_true, z_pred1, z_pred2, z_pred3)

###########################################################################################

# 误差分析: 
plot_error_distribution(z_true, z_pred1, z_pred2, z_pred3)

###########################################################################################

# 统计指标
print_error_statistic(z_true, z_pred1, z_pred2, z_pred3)
