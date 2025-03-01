import numpy as np
import matplotlib.pyplot as plt

def plot_signals(x_, y_, z_, x, y, z):
    # 显示原始信号数据和加噪声的信号数据chaos 图像
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot(x_, y_, z_, lw=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title('Original Lorenz Attractor')

    ax = fig.add_subplot(122, projection='3d')
    ax.plot(x, y, z, lw=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title('Noised Lorenz Attractor')
    plt.show()


def plot_training_process(train_losses, test_losses):
    # 可视化训练过程
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


######################################################################################
def plot_truth_vs_prediction(z_true, z_pred1, z_pred2, z_pred3):
    fig = plt.figure(figsize=(8, 12))
    # 可视化预测结果: x,y -> z
    fig.add_subplot(311)
    plt.plot(z_true[:1000, 2], label='True Values', alpha=0.7)
    plt.plot(z_pred1[:1000, 2], label='Predictions', alpha=0.7)
    plt.xlabel('Time Step')
    plt.ylabel('z Value')
    plt.title('True vs Predicted z Values')
    plt.legend()


    # 可视化预测结果:x,z -> y
    fig.add_subplot(312)
    plt.plot(z_true[:1000, 1], label='True Values', alpha=0.7)
    plt.plot(z_pred2[:1000, 1], label='Predictions', alpha=0.7)
    plt.xlabel('Time Step')
    plt.ylabel('y Value')
    plt.title('True vs Predicted y Values')
    plt.legend()


    # 可视化预测结果:y,z -> x
    fig.add_subplot(313)
    plt.plot(z_true[:1000, 0], label='True Values', alpha=0.7)
    plt.plot(z_pred3[:1000, 0], label='Predictions', alpha=0.7)
    plt.xlabel('Time Step')
    plt.ylabel('x Value')
    plt.title('True vs Predicted x Values')
    plt.legend()


    plt.show()

###########################################################################################

# 误差分析: x,y -> z
def plot_error_distribution(z_true, z_pred1, z_pred2, z_pred3):
    errors = z_true - z_pred1

    plt.figure(figsize=(12, 6))
    plt.subplot(3, 2, 1)
    plt.scatter(z_true, z_pred1, s=5, alpha=0.5)
    plt.plot([z_true.min(), z_true.max()], [z_true.min(), z_true.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Prediction vs True Value')

    plt.subplot(3, 2, 2)
    plt.hist(errors, bins=50)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.tight_layout()

    # 误差分析: x,z -> y
    errors = z_true - z_pred2

    plt.subplot(3, 2, 3)
    plt.scatter(z_true, z_pred2, s=5, alpha=0.5)
    plt.plot([z_true.min(), z_true.max()], [z_true.min(), z_true.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Prediction vs True Value')

    plt.subplot(3, 2, 4)
    plt.hist(errors, bins=50)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.tight_layout()

    # 误差分析: y,z -> x
    errors = z_true - z_pred3

    plt.subplot(3, 2, 5)
    plt.scatter(z_true, z_pred3, s=5, alpha=0.5)
    plt.plot([z_true.min(), z_true.max()], [z_true.min(), z_true.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Prediction vs True Value')

    plt.subplot(3, 2, 6)
    plt.hist(errors, bins=50)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.tight_layout()


    plt.show()

###########################################################################################

# 统计指标: x,y -> z
def print_error_statistic(z_true, z_pred1, z_pred2, z_pred3):
    errors = z_true - z_pred1

    mse = np.mean(errors**2)
    mae = np.mean(np.abs(errors))
    print(f"Mean Squared Error of z: {mse:.4f}")
    print(f"Mean Absolute Error of z: {mae:.4f}")
    print(f"Max Absolute Error of z: {np.max(np.abs(errors)):.4f}")

    # 统计指标: x,z -> y
    errors = z_true - z_pred2

    mse = np.mean(errors**2)
    mae = np.mean(np.abs(errors))
    print(f"Mean Squared Error of y: {mse:.4f}")
    print(f"Mean Absolute Error of y: {mae:.4f}")
    print(f"Max Absolute Error of y: {np.max(np.abs(errors)):.4f}")

    # 统计指标: y,z -> x
    errors = z_true - z_pred3

    mse = np.mean(errors**2)
    mae = np.mean(np.abs(errors))
    print(f"Mean Squared Error of x: {mse:.4f}")
    print(f"Mean Absolute Error of x: {mae:.4f}")
    print(f"Max Absolute Error of x: {np.max(np.abs(errors)):.4f}")

