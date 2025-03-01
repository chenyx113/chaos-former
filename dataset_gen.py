#dataset_gen.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from display_signal import plot_signals

# 生成Lorenz数据
def generate_lorenz_data():
    sigma = 10.0
    rho = 28.0
    beta = 8.0/3
    y0 = [1.0, 1.0, 1.0]
    h = 0.01
    t_end = 100

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
    xyz = np.column_stack((x, y, z))
    xyz_normalized = scaler.fit_transform(xyz)

    X = []
    Z = []
    for i in range(window_size, len(xyz_normalized)):
        X.append(xyz_normalized[i-window_size:i])
        Z.append(xyz_normalized[i])
    
    X = np.array(X)
    Z = np.array(Z)
    return X, Z, scaler


# 添加噪声干扰
def add_datanoise(x, y, z):
    x_noise = np.random.rand(x.shape[0])
    y_noise = np.random.rand(y.shape[0]) 
    z_noise = np.random.rand(z.shape[0]) 
    x = x + x_noise
    y = y + y_noise
    z = z + z_noise

    return x,y,z

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

def generate_signal():
    x_, y_, z_ = generate_lorenz_data()
    x, y, z = add_datanoise(x_, y_, z_)
    plot_signals(x_, y_, z_, x, y, z)
    return x,y,z

