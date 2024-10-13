import numpy as np

from helpers import plot_velocities
from functools import partial
import matplotlib.pyplot as plt

UV = np.load('data/UV.npy')
x = np.load('data/x.npy')
y = np.load('data/y.npy')

n = UV.shape[1] * UV.shape[2] * UV.shape[3]
m = UV.shape[0]
X = np.zeros((n, m))

u_min = np.min(UV[:, :, :, 0])
u_max = np.max(UV[:, :, :, 0])
v_min = np.min(UV[:, :, :, 1])
v_max = np.max(UV[:, :, :, 1])
plot_vel = partial(plot_velocities, x=x, y=y, n=n)
for i in range(UV.shape[0]):
    X[:n // 2, i] = UV[i, :, :, 0].ravel()
    X[n // 2:, i] = UV[i, :, :, 1].ravel()

# get pod modes
svd = np.linalg.svd(X, full_matrices=False)
U = svd[0]
S = np.diag(svd[1])
V = svd[2].T

# log plot singular values
plt.scatter(range(S.shape[0]), np.log(S.diagonal()))
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()

# plot first k POD modes
k = 20
for i in range(k):
    plot_vel(U[:, i])
    print(S[i, i])
    # save plot
    plt.savefig(f'plots/pod_mode_{i}.png')
# todo there is something *up* with modes 12, 13, 15, 16
