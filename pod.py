import numpy as np

from helpers import plot_velocities, read_data
from functools import partial
import matplotlib.pyplot as plt

X, X_norm, n, m, x, y, UV, u_min, u_max, v_min, v_max = read_data()
plot_vel = partial(plot_velocities, x=x, y=y, n=n)

# get pod modes
svd = np.linalg.svd(X, full_matrices=False)
U = svd[0]
S = np.diag(svd[1])
V = svd[2].T

# log plot singular values
plt.scatter(range(S.shape[0]), np.log10(S.diagonal()))
plt.xlabel('Index', fontsize=14)
plt.ylabel(r'log $\sigma$', fontsize=14)
plt.show()

# plot first k POD modes
k = 50
for i in range(k):
    plot_vel(U[:, i])
    plt.savefig(f'plots/pod_mode_{i}.png')
    plt.close()
    plt.plot(V[:, i])
    plt.xlabel('$t$', fontsize=14)
    plt.ylabel(f'$v_{{{i}}}$', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'plots/pod_mode_{i}_v.png')
    plt.close()
    # plot time evolution
    plt.plot(V[:, i])
    plt.show()
