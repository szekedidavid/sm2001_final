import numpy as np  # todo does DMD work on a chaotic system...

from helpers import plot_velocities, read_data
from functools import partial
import matplotlib.pyplot as plt

X, n, m, UV, x, y = read_data()

u_min = np.min(UV[:, :, :, 0])
u_max = np.max(UV[:, :, :, 0])
v_min = np.min(UV[:, :, :, 1])
v_max = np.max(UV[:, :, :, 1])

plot_vel = partial(plot_velocities, x=x, y=y, n=n)
X_norm = np.zeros_like(X)
X_norm[:n // 2] = X[:n // 2] - np.mean(X[:n // 2])
X_norm[n // 2:] = X[n // 2:] - np.mean(X[n // 2:])

# get dmd modes
X_k = X_norm[:, :-1]
X_kp1 = X_norm[:, 1:]
U, S, Vt = np.linalg.svd(X_k, full_matrices=False)

# log plot singular values
plt.scatter(range(S.shape[0]), np.log(S))
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()


r = 900 # too small gives a result that decays, too large gives a result that explodes
U_red = U[:, :r]
S_red = np.diag(S[:r])
V_red = Vt.T[:, :r]
# A_red = U_red.T @ X_kp1 @ V_red @ np.linalg.inv(S_red)
A_red = np.linalg.solve(S_red.T, (U_red.T @ X_kp1 @ V_red).T).T

lambd, W_red = np.linalg.eig(A_red)
# Phi = X_kp1 @ V_red @ np.linalg.inv(S_red) @ W_red
Phi = X_kp1 @ np.linalg.solve(S_red.T, V_red.T).T @ W_red

# ritz plot of eigenvalues
plt.scatter(np.real(lambd), np.imag(lambd))
theta = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(theta), np.sin(theta), color='black')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.show()

C = np.zeros((r, m - 1), dtype=np.complex128)
for i, lambd_val in enumerate(lambd):
    C[i] = lambd_val ** np.arange(0, m - 1)

P = ((W_red.conj().T @ W_red) * (C @ C.conj().T)).conj()
p = np.diag(C @ V_red @ S_red @ W_red).conj()
b = np.linalg.solve(P, p)
# b = np.linalg.pinv(Phi) @ X_k[:, 0]


# reconstruct UV using DMD modes
X_dmd = Phi @ np.diag(b) @ C

# plot reconstructed UV
times = 0, 5, 50, 100, 300, 500, 900
for i in times:
    plot_vel(X_norm[:, i], u_min=u_min, u_max=u_max, v_min=v_min, v_max=v_max)
    plot_vel(np.real(X_dmd[:, i]), u_min=u_min, u_max=u_max, v_min=v_min, v_max=v_max)

# print RMS
rms = np.sqrt(np.mean((X_k - np.real(X_dmd)) ** 2))
print(f'RMS: {rms}')