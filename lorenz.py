import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义Lorenz系统的微分方程
def lorenz_deriv(t, xyz, sigma, rho, beta):
    x, y, z = xyz
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return np.array([dx_dt, dy_dt, dz_dt])

# 四阶龙格-库塔法积分器
def rk4_integrate(func, y0, t0, t_end, h, **kwargs):
    steps = int(np.ceil((t_end - t0) / h))
    t = np.zeros(steps + 1)
    y = np.zeros((steps + 1, len(y0)))
    t[0] = t0
    y[0] = y0
    for i in range(steps):
        k1 = func(t[i], y[i], **kwargs)
        k2 = func(t[i], y[i] + k1 * h/2, **kwargs)
        k3 = func(t[i], y[i] + k2 * h/2, **kwargs)
        k4 = func(t[i] + h, y[i] + k3 * h, **kwargs)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) * h / 6
        t[i+1] = t[i] + h
    return t, y

# 参数设置
sigma = 10.0
beta = 8.0 / 3

# ==========================
# 1. Hopf分岔图（参数rho变化）
# ==========================
rho_min = 20
rho_max = 30
num_rho = 200  # 减少数值以加快计算
rho_values = np.linspace(rho_min, rho_max, num_rho)
h = 0.01
t_transient = 100  # 瞬态排除时间
t_keep = 50       # 稳态保留时间

bifurcation_rho = []
bifurcation_x = []

for rho in rho_values:
    y0 = np.array([1.0, 1.0, 1.0])
    t, y = rk4_integrate(lorenz_deriv, y0, 0, t_transient + t_keep, h,
                        sigma=sigma, rho=rho, beta=beta)
    x = y[int(t_transient/h):, 0]  # 排除瞬态
    # 寻找局部极大值
    maxima = (np.diff(np.sign(np.diff(x)))) < 0
    maxima_x = x[1:-1][maxima]
    bifurcation_rho.extend([rho] * len(maxima_x))
    bifurcation_x.extend(maxima_x)

plt.figure(figsize=(10, 6))
plt.scatter(bifurcation_rho, bifurcation_x, s=0.5, c='k', alpha=0.5)
plt.xlabel('rho')
plt.ylabel('x maxima')
plt.title('Bifurcation Diagram (rho vs x maxima)')
plt.show()

# ==========================
# 2. 庞加莱截面图（z=rho-1平面）
# ==========================
rho_p = 28.0
z_cross = rho_p - 1  # 平衡点对应的z值
y0 = np.array([1.0, 1.0, 1.0])
h = 0.01
t_end = 100
t, y = rk4_integrate(lorenz_deriv, y0, 0, t_end, h,
                    sigma=sigma, rho=rho_p, beta=beta)

crossing_points = []
for i in range(len(y)-1):
    z_prev = y[i, 2]
    z_current = y[i+1, 2]
    if z_prev < z_cross and z_current >= z_cross:  # 向上穿过截面
        alpha = (z_cross - z_prev) / (z_current - z_prev)
        x_interp = y[i, 0] + alpha * (y[i+1, 0] - y[i, 0])
        y_interp = y[i, 1] + alpha * (y[i+1, 1] - y[i, 1])
        crossing_points.append([x_interp, y_interp])

crossing_points = np.array(crossing_points)

plt.figure(figsize=(8, 8))
plt.scatter(crossing_points[:, 0], crossing_points[:, 1], s=5, alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Poincaré Section (z={z_cross}, rho={rho_p})')
plt.show()

# ==========================
# 3. 混沌吸引子三维相图
# ==========================
t_end = 100
t, y = rk4_integrate(lorenz_deriv, y0, 0, t_end, h,
                    sigma=sigma, rho=rho_p, beta=beta)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot(y[:, 0], y[:, 1], y[:, 2], lw=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title('Lorenz Attractor')
plt.show()
