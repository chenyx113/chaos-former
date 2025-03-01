#eval_signal_onnx.py

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from dataset_gen import *
from display_signal import *
from models import TransformerModel
import onnx
import onnxruntime as ort

# 训练参数设置
window_size = 32
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
input_data = X_test[0,:,:].reshape(1, window_size, signal_dim)
input_data = torch.FloatTensor(input_data)

test_pred1 = model_best(input_data)
torch.onnx.export(model_best, input_data, 'best_model.onnx', input_names=['input'], output_names=['output'])

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



ort_session = ort.InferenceSession('best_model.onnx')

test_pred1 = []
test_pred2 = []
test_pred3 = []


with torch.no_grad():
    for i in range(X_test.shape[0]):
        input_data = X_test[i, :, :].reshape(1, window_size, signal_dim)
        X_test1 = np.copy(input_data)
        X_test2 = np.copy(input_data)
        X_test3 = np.copy(input_data)

        X_test1[:, :, 2] = 0.0
        X_test2[:, :, 1] = 0.0
        X_test3[:, :, 0] = 0.0
        #print(X_test1)
        #print('##################################')

        X_test1 = torch.FloatTensor(X_test1)
        X_test2 = torch.FloatTensor(X_test2)
        X_test3 = torch.FloatTensor(X_test3)
        #print(X_test1)
        #print('##################################')

        ort_input1 = {ort_session.get_inputs()[0].name:X_test1.numpy()}
        ort_input2 = {ort_session.get_inputs()[0].name:X_test2.numpy()}
        ort_input3 = {ort_session.get_inputs()[0].name:X_test3.numpy()}
        '''
        test_pred1.append(ort_session.run(None, ort_input1))
        test_pred2.append(ort_session.run(None, ort_input2))
        test_pred3.append(ort_session.run(None, ort_input3))
        '''
        test_pred1.append(model_best(X_test1).numpy())
        test_pred2.append(model_best(X_test2).numpy())
        test_pred3.append(model_best(X_test3).numpy())


test_pred1 = np.array(test_pred1)
test_pred2 = np.array(test_pred2)
test_pred3 = np.array(test_pred3)

dims = test_pred1.shape
print(dims)

test_pred1 = test_pred1.reshape(dims[0], dims[2])
test_pred2 = test_pred2.reshape(dims[0], dims[2])
test_pred3 = test_pred3.reshape(dims[0], dims[2])
print(test_pred1.shape)

z_pred1 = scaler.inverse_transform(test_pred1)
z_pred2 = scaler.inverse_transform(test_pred2)
z_pred3 = scaler.inverse_transform(test_pred3)
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
