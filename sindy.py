import numpy as np

from helpers import plot_velocities, read_data
from functools import partial
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pysindy as ps
from pysindy import optimizers
import pickle


X, X_norm, n, m, x, y, UV, u_min, u_max, v_min, v_max = read_data()
plot_vel = partial(plot_velocities, x=x, y=y, n=n)

svd = np.linalg.svd(X_norm, full_matrices=False)
U = svd[0]
S = np.diag(svd[1])
V = svd[2].T

r = 10
max_iter = 50
U_red = U[:, :r]
S_red = S[:r, :r]
V_red = V[:, :r]

X_train, X_test, V_train, V_test = train_test_split(X_norm.T, V_red, test_size=0.2, shuffle=False)

coef_train = (S_red @ V_train.T).T
coef_test = (S_red @ V_test.T).T

scaler = StandardScaler()
coef_train = scaler.fit_transform(coef_train)
coef_test = scaler.transform(coef_test)

# lambd = 0.001
# eta = 1e20
threshold = 0.0001
alpha = 0.5

poly_lib = ps.PolynomialLibrary(degree=2, include_bias=False)
# combine libraries
library = poly_lib
optimizer = ps.STLSQ(threshold=threshold)
# optimizer = ps.TrappingSR3(threshold=lambd, eta=eta, thresholder='L1', max_iter=max_iter, verbose=True)
model = ps.SINDy(
    optimizer=optimizer,
    feature_library=library,
)

V_red_dot = np.diff(coef_train, axis=0)
V_red_dot = np.vstack([V_red_dot, V_red_dot[-1, :]])
model.fit(coef_train, x_dot=V_red_dot)
model.print()


Xi = model.coefficients().T
# save Xi
# np.save(f'Xi_{r}_{lambd}_new.npy', Xi)

# load everything
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
# upper bound 0.4, lower -0.4
cax = ax.matshow(Xi, cmap='coolwarm', aspect='auto', vmin=-0.4, vmax=0.4)
fig.colorbar(cax)
plt.tight_layout()
plt.show()

V_dot_sindy = model.predict(coef_train)
#
# # print error in predicting V_dot
mse_dot = np.mean((V_dot_sindy - V_red_dot) ** 2)
print(f'MSE V_dot: {mse_dot}')
#
# # print training error
V_sindy_train = model.simulate(coef_train[0], np.arange(0, coef_train.shape[0]), integrator='solve_ivp', integrator_kws={'method': 'LSODA'})
X_sindy_train = U_red @ S_red @ V_sindy_train.T
#
# save
# np.save(f'V_sindy_train_{r}_{lambd}.npy', V_sindy_train)
# np.save(f'X_sindy_train_{r}_{lambd}.npy', X_sindy_train)

# print test error
V_sindy_test = model.simulate(coef_test[0], np.arange(0, coef_test.shape[0]), integrator='solve_ivp', integrator_kws={'method': 'LSODA'})
X_sindy_test = U_red @ S_red @ V_sindy_test.T

# save
# np.save(f'V_sindy_test_{r}_{lambd}.npy', V_sindy_test)
# np.save(f'X_sindy_test_{r}_{lambd}.npy', X_sindy_test)

mse_x_time = np.mean((X_sindy_train - X_train.T) ** 2, axis=0)
plt.plot(np.arange(0, 800), mse_x_time)
plt.xlabel('$t$')
plt.ylabel('MSE')
plt.xlim(0, 800)
plt.show()
mse_x = np.mean(mse_x_time)
print(f'MSE train: {mse_x}')

mse_test_time = np.mean((X_sindy_test - X_test.T) ** 2, axis=0)
plt.plot(np.arange(800, 1000), mse_test_time)
plt.xlabel('$t$')
plt.ylabel('MSE')
plt.xlim(800, 1000)
plt.show()
mse_test = np.mean(mse_test_time)
print(f'MSE test: {mse_test}')

# print test POD error
V_pod_test = U_red @ S_red @ coef_test.T
mse_pod_test = np.mean((V_pod_test - X_test.T) ** 2)
print(f'MSE POD test: {mse_pod_test}')

# plot real and predicted time series
for i in range(15):
    times = np.arange(800, 1000)
    plt.plot(times, V_sindy_test.T[i], label='Predicted')
    plt.plot(times, coef_test.T[i], label='Real')
    plt.xlim(800, 1000)
    plt.xlabel('$t$')
    plt.ylabel(f'$V_{{{i}}}$')
    plt.legend()
    plt.show()
