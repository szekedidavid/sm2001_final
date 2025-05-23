import numpy as np
import matplotlib.pyplot as plt

def read_data():
    UV = np.load('data/UV.npy')
    x = np.load('data/x.npy')
    y = np.load('data/y.npy')

    n = UV.shape[1] * UV.shape[2] * UV.shape[3]
    m = UV.shape[0]
    X = np.zeros((n, m))
    for i in range(UV.shape[0]):
        X[:n // 2, i] = UV[i, :, :, 0].ravel()
        X[n // 2:, i] = UV[i, :, :, 1].ravel()
    X_norm = np.zeros_like(X)
    X_norm[:n // 2] = X[:n // 2] - np.mean(X[:n // 2])
    X_norm[n // 2:] = X[n // 2:] - np.mean(X[n // 2:])
    u_min = np.min(UV[:, :, :, 0])
    u_max = np.max(UV[:, :, :, 0])
    v_min = np.min(UV[:, :, :, 1])
    v_max = np.max(UV[:, :, :, 1])

    return X, X_norm, n, m, x, y, UV, u_min, u_max, v_min, v_max


def plot_velocities(vel, x, y, n, u_min=None, u_max=None, v_min=None, v_max=None):
    if vel.shape[0] == n:
        U_vel = vel[:n // 2]
        V_vel = vel[n // 2:]
        U_vel_grid = U_vel.reshape(x.shape[0], x.shape[1])
        V_vel_grid = V_vel.reshape(x.shape[0], x.shape[1])
    else:
        U_vel_grid = vel[:, :, 0]
        V_vel_grid = vel[:, :, 1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
    c_u = ax1.pcolormesh(x, y, U_vel_grid, cmap='coolwarm', vmin=u_min, vmax=u_max)
    ax1.set_aspect('equal')
    fig.colorbar(c_u, ax=ax1, orientation='horizontal')
    ax1.set_title('$u$ component')
    c_v = ax2.pcolormesh(x, y, V_vel_grid, cmap='coolwarm', vmin=v_min, vmax=v_max)
    ax2.set_aspect('equal')
    fig.colorbar(c_v, ax=ax2, orientation='horizontal')
    ax2.set_title('$v$ component')
    plt.tight_layout()
