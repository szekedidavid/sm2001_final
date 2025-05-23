import numpy as np

from helpers import plot_velocities, read_data
from functools import partial
import matplotlib.pyplot as plt

X, X_norm, n, m, x, y, UV, u_min, u_max, v_min, v_max = read_data()
plot_vel = partial(plot_velocities, x=x, y=y, n=n)

# get pod modes
X_k = X_norm[:, :-1]
X_kp1 = X_norm[:, 1:]
U, S, Vt = np.linalg.svd(X_k, full_matrices=False)
V = Vt.T

# log plot singular values
plt.scatter(range(S.shape[0]), np.log(S))
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()

r_max = 900 # X_k.shape[1]
r = r_max

U_red = U[:, :r]
S_red = np.diag(S[:r])
V_red = Vt.T[:, :r]
A_red = np.linalg.solve(S_red.T, (U_red.T @ X_kp1 @ V_red).T).T

lambd, W_red = np.linalg.eig(A_red)
Phi = X_kp1 @ np.linalg.solve(S_red.T, V_red.T).T @ W_red

# ritz plot of eigenvalues
plt.scatter(np.real(lambd), np.imag(lambd))
theta = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(theta), np.sin(theta), color='black')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.xlabel('Re $\lambda$', fontsize=18)
plt.ylabel('Im $\lambda$', fontsize=18)
plt.xticks([-1.00, -0.50, 0.00, 0.50, 1.00], fontsize=14)
plt.yticks([-1.00, -0.50, 0.00, 0.50, 1.00], fontsize=14)
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

C = np.zeros((r, m - 1), dtype=np.complex128)
for i, lambd_val in enumerate(lambd):
    C[i] = lambd_val ** np.arange(0, m - 1)

P = (W_red.conj().T @ W_red) * (C @ C.conj().T).conj()
p = np.diag(C @ V_red @ S_red.conj().T @ W_red).conj()
b = np.linalg.solve(P, p)
# b = np.linalg.pinv(Phi) @ X_k[:, 0]

# get one more column in C
C_full = np.zeros((r, m), dtype=np.complex128)
C_full[:, :-1] = C
C_full[:, -1] = lambd ** (m - 1)

# reconstruct UV using DMD modes
X_dmd = Phi @ np.diag(b) @ C_full

# get energy from each DMD mode
energy_list = []
for i in range(r):
    x = np.outer(Phi[:, i], C_full[i]) * b[i]
    energy = np.linalg.norm(x) ** 2
    energy_list.append(energy)
energy_list = np.array(energy_list)
sorted_indices = np.argsort(energy_list)[::-1]
sorted_energy_list = energy_list[sorted_indices]
Phi_sorted = Phi[:, sorted_indices]

# plot first k DMD modes
k = 20
for i in range(k):
    plot_vel(np.real(Phi_sorted[:, i]))
    plt.savefig(f'./plots/dmd_mode_{i}_real.png')
    plot_vel(np.imag(Phi_sorted[:, i]))
    plt.savefig(f'./plots/dmd_mode_{i}_imag.png')

    # print eigenvalue
    print(f'Eigenvalue {i}: {lambd[sorted_indices[i]]}')
    plt.show()


# plot reconstructed UV
# times = 0, 300, 600, 900
# for i in times:
#     plot_vel(X_norm[:, i], u_min=u_min, u_max=u_max, v_min=v_min, v_max=v_max)
#     plt.show()
#     plot_vel(np.real(X_dmd[:, i]), u_min=u_min, u_max=u_max, v_min=v_min, v_max=v_max)
#     plt.show()

# plot MSE with time
mse = np.mean((X_norm - np.real(X_dmd)) ** 2, axis=0)
plt.plot(mse)
plt.xlabel('$t$')
plt.ylabel('MSE')
plt.tight_layout()
plt.show()

# print POD MSE
X_pod = U_red @ S_red @ V_red.T
mse_pod = np.mean((X_norm - X_pod) ** 2)
print(f'MSE POD: {mse_pod}')

# print MSE
mse = np.mean((X_norm - np.real(X_dmd)) ** 2)
print(f'MSE DMD: {mse}')
