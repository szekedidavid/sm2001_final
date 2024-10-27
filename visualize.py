import numpy as np

from helpers import plot_velocities, read_data
from functools import partial
import matplotlib.pyplot as plt

X, X_norm, n, m, x, y, UV, u_min, u_max, v_min, v_max = read_data()
plot_vel = partial(plot_velocities, x=x, y=y, n=n, u_min=u_min, u_max=u_max, v_min=v_min, v_max=v_max)

for i in [0, 600]:
    plot_vel(X[:, i], u_min=u_min, u_max=u_max, v_min=v_min, v_max=v_max)
    plt.show()

# mean velocity (u and v) over x and time
U_mean = np.mean(UV[:, :, :, 0], axis=(0, 1))
V_mean = np.mean(UV[:, :, :, 1], axis=(0, 1))

# reynolds stress
u_fluct = UV[:, :, :, 0] - U_mean
v_fluct = UV[:, :, :, 1] - V_mean
reynolds_stress = np.mean(u_fluct * v_fluct, axis=(0, 1))
U_var = np.mean(u_fluct ** 2, axis=(0, 1))

# plot U_mean vs y
plt.plot(U_mean, y[0])
plt.ylim(y[0][0], y[0][-1])
plt.xlim(0, 1.2)
plt.xlabel('$\overline{u}$', fontsize=18)
plt.ylabel('$y$', fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# plot U_var vs y
plt.plot(U_var, y[0])
plt.ylim(y[0][0], y[0][-1])
plt.xlim(0, 0.11)
plt.xlabel(r"$\overline{u'^2}$", fontsize=18)
plt.ylabel('$y$', fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# plot reynolds stress vs y
plt.plot(reynolds_stress, y[0])
plt.ylim(y[0][0], y[0][-1])
# plt.xlim(-0.011, 0.011)
plt.axvline(0, color='black', linestyle='--')
plt.xlabel(r"$\overline{u'v'}$", fontsize=18)
plt.ylabel('$y$', fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# plot V_var vs y
V_var = np.mean(v_fluct ** 2, axis=(0, 1))
plt.plot(V_var, y[0])
plt.ylim(y[0][0], y[0][-1])
plt.xlim(0, 0.14)
plt.xlabel(r"$\overline{v'^2}$", fontsize=18)
plt.ylabel('$y$', fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
