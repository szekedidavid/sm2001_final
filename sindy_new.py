import numpy as np

from helpers import plot_velocities, read_data
from functools import partial
import matplotlib.pyplot as plt
import matplotlib
import pysindy as ps

X, X_norm, n, m, x, y, UV, u_min, u_max, v_min, v_max = read_data()
plot_vel = partial(plot_velocities, x=x, y=y, n=n)

svd = np.linalg.svd(X_norm, full_matrices=False)
U = svd[0]
S = np.diag(svd[1])
V = svd[2].T

r = 20
U_red = U[:, :r]
S_red = S[:r, :r]
V_red = V[:, :r]

tau = 0.01
# create library
library = ps.PolynomialLibrary(degree=2)
optimizer = ps.STLSQ(threshold=tau)
model = ps.SINDy(
    optimizer=optimizer,
    feature_library=library,
)
model.fit(V_red, t=1)
model.print()

times = np.arange(0, V_red.shape[0])
# reconstruct flow
Xi = model.coefficients().T

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
cax = ax.matshow(Xi, cmap='coolwarm', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=tau))
fig.colorbar(cax)
plt.show()



V_sindy = np.zeros_like(V_red)
V_sindy[0] = V_red[0]
for i in range(1, m):
    V_sindy[i] = V_sindy[i - 1] + model.predict(V_sindy[i - 1].T).T
X_sindy = U_red @ S_red @ V_sindy.T

# print min MSE from POD
X_pod = U_red @ S_red @ V_red.T
mse_pod = np.mean((X_norm - X_pod) ** 2)
print(f'MSE POD: {mse_pod}')

# print MSE
mse = np.mean((X_norm - X_sindy) ** 2)
print(f'MSE SINDy: {mse}')