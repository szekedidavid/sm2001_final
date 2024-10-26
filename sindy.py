import numpy as np

from helpers import plot_velocities, read_data
from functools import partial
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
import pysindy as ps
from pysindy import optimizers


X, X_norm, n, m, x, y, UV, u_min, u_max, v_min, v_max = read_data()
plot_vel = partial(plot_velocities, x=x, y=y, n=n)

svd = np.linalg.svd(X_norm, full_matrices=False)
U = svd[0]
S = np.diag(svd[1])
V = svd[2].T

r = 12
U_red = U[:, :r]
S_red = S[:r, :r]
V_red = V[:, :r]

X_train, X_test, V_train, V_test = train_test_split(X_norm.T, V_red, test_size=0.2, shuffle=False)

lambd = 0.01
eta = 0.1

poly_lib = ps.PolynomialLibrary(degree=2, include_bias=False)
library = poly_lib
# optimizer = ps.STLSQ(threshold=threshold, alpha=alpha)
optimizer = ps.TrappingSR3(threshold=lambd, eta=eta, thresholder='L1', max_iter=10)
model = ps.SINDy(
    optimizer=optimizer,
    feature_library=library,
)

V_red_dot = np.diff(V_train, axis=0)
V_red_dot = np.vstack([V_red_dot, V_red_dot[-1, :]])
model.fit(V_train, x_dot=V_red_dot)
model.print()

Xi = model.coefficients().T

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
cax = ax.matshow(Xi, cmap='coolwarm', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=0.1))
fig.colorbar(cax)
plt.show()

V_dot_sindy = model.predict(V_train)

# print error in predicting V_dot
mse_dot = np.mean((V_dot_sindy - V_red_dot) ** 2)
print(f'MSE V_dot: {mse_dot}')

# print training error
# V_sindy_train = model.simulate(V_train[0], np.arange(0, V_train.shape[0]), integrator='solve_ivp', integrator_kws={'method': 'Radau'})
# X_sindy_train = U_red @ S_red @ V_sindy_train.T
#
# mse_x = np.mean((X_sindy_train - X_train.T) ** 2)
# print(f'MSE train: {mse_x}')

# print test error
V_sindy_test = model.simulate(V_test[0], np.arange(0, V_test.shape[0]), integrator='solve_ivp', integrator_kws={'method': 'Radau'})
X_sindy_test = U_red @ S_red @ V_sindy_test.T

mse_test = np.mean((X_sindy_test - X_test.T) ** 2)
print(f'MSE test: {mse_test}')

# print test POD error
V_pod_test = U_red @ S_red @ V_test.T
mse_pod_test = np.mean((V_pod_test - X_test.T) ** 2)
print(f'MSE POD test: {mse_pod_test}')
