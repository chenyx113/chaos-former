#eval_signal.py

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from dataset_gen import *
from display_signal import *
from models import TransformerModel


# 训练参数设置
window_size = 32
learning_rate = 1e-5
signal_dim = 3

# 生成并预处理数据
x,y,z = generate_signal()

# use x,y to fit z
X, Z, scaler = prepare_data(x, y, z, window_size)

# 划分训练集和测试集
X_train, X_test, Z_train, Z_test = train_test_split(
    X, Z, test_size=0.2, shuffle=False)

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
