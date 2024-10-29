import numpy as np

from helpers import plot_velocities, read_data
from functools import partial
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pysindy as ps
import matplotlib

# pysindy models
import pysindy.optimizers as optimizers


X, X_norm, n, m, x, y, UV, u_min, u_max, v_min, v_max = read_data()
plot_vel = partial(plot_velocities, x=x, y=y, n=n)

svd = np.linalg.svd(X_norm, full_matrices=False)
U = svd[0]
S = np.diag(svd[1])
V = svd[2].T

r = 10
max_iter = 20
U_red = U[:, :r]
S_red = S[:r, :r]
V_red = V[:, :r]

X_train, X_test, V_train, V_test = train_test_split(X_norm.T, V_red, test_size=0.2, shuffle=False)

scaler = StandardScaler()
V_train_norm = scaler.fit_transform(V_train)
V_test_norm = scaler.transform(V_test)

lambd = 6
eta = 2
# threshold = 0.015  # 0.05 decent
# alpha = 0.5

poly_lib = ps.PolynomialLibrary(degree=2, include_bias=False)
# optimizer = ps.STLSQ(threshold=threshold, alpha=alpha, verbose=True)
optimizer = ps.TrappingSR3(threshold=lambd, eta=eta, thresholder='L1', max_iter=max_iter, verbose=True)

model = ps.SINDy(
    optimizer=optimizer,
    feature_library=poly_lib,
)

V_red_dot = np.diff(V_train_norm, axis=0)
V_red_dot = np.vstack([V_red_dot, V_red_dot[-1, :]])
model.fit(V_train_norm)
model.print()

Xi = model.coefficients().T
# save Xi
# np.save(f'Xi_{r}_{lambd}.npy', Xi)

# # load everything
# Xi = np.load(f'Xi_{r}_{lambd}_new.npy')
# V_sindy_train = np.load(f'V_sindy_train_{r}_{lambd}.npy')
# X_sindy_train = np.load(f'X_sindy_train_{r}_{lambd}.npy')
# V_sindy_test = np.load(f'V_sindy_test_{r}_{lambd}.npy')
# X_sindy_test = np.load(f'X_sindy_test_{r}_{lambd}.npy')

# give sparsity ratio of Xi
tau = 1e-5
Xi[np.abs(Xi) < tau] = 0
sparsity_ratio = np.count_nonzero(Xi) / Xi.size
print(f'Sparsity ratio: {sparsity_ratio}')

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
cax = ax.matshow(Xi, cmap='coolwarm', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=1e-3, linscale=1))
fig.colorbar(cax)
plt.tight_layout()
plt.show()


# print test error
V_sindy_test_norm = model.simulate(V_test_norm[0], np.arange(0, V_test.shape[0]), integrator='solve_ivp', integrator_kws={'method': 'LSODA'})
V_sindy_test = scaler.inverse_transform(V_sindy_test_norm)
X_sindy_test = U_red @ S_red @ V_sindy_test.T

mse_test_time = np.mean((X_sindy_test - X_test.T) ** 2, axis=0)
plt.plot(np.arange(800, 1000), mse_test_time)
plt.xlabel('$t$')
plt.ylabel('MSE')
plt.xlim(800, 1000)
plt.show()
mse_test = np.mean(mse_test_time)
print(f'MSE test: {mse_test}')

fig, axes = plt.subplots(5, 3, figsize=(10, 11), sharex=True)
fig.subplots_adjust(hspace=0.5, wspace=0.4)
times = np.arange(800, 1000)
for i in range(r):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    ax.plot(times, V_sindy_test.T[i], label='Predicted')
    ax.plot(times, V_test.T[i], label='Real')
    ax.set_xlim(800, 1000)
    ax.set_xlabel('$t$')
    ax.set_ylabel(f'$V_{{{i}}}$')
    if i == 0:
        ax.legend()
plt.tight_layout()
plt.show()
# save
np.save(f'V_sindy_test_{r}_{lambd}.npy', V_sindy_test)
np.save(f'X_sindy_test_{r}_{lambd}.npy', X_sindy_test)

# print training error
# # V_sindy_train_norm = model.simulate(V_train_norm[0], np.arange(0, V_train.shape[0]), integrator='solve_ivp', integrator_kws={'method': 'LSODA'})
# V_sindy_train = scaler.inverse_transform(V_sindy_train_norm)
# X_sindy_train = U_red @ S_red @ V_sindy_train.T
# mse_x_time = np.mean((X_sindy_train - X_train.T) ** 2, axis=0)
# plt.plot(np.arange(0, 800), mse_x_time)
# plt.xlabel('$t$')
# plt.ylabel('MSE')
# plt.xlim(0, 800)
# plt.show()
# mse_x = np.mean(mse_x_time)
# print(f'MSE train: {mse_x}')
