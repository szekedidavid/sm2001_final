import numpy as np

from helpers import plot_velocities, read_data
from functools import partial
import matplotlib.pyplot as plt

X, X_norm, n, m, x, y, UV, u_min, u_max, v_min, v_max = read_data()
plot_vel = partial(plot_velocities, x=x, y=y, n=n, u_min=u_min, u_max=u_max, v_min=v_min, v_max=v_max)

# plot first data point
for i in [0, 600]:
    plot_vel(X[:, i], u_min=u_min, u_max=u_max, v_min=v_min, v_max=v_max)

# Mean velocity (u and v) over x and time
U_mean = np.mean(UV[:, :, :, 0], axis=(0, 1))  # Mean of u over time and x
V_mean = np.mean(UV[:, :, :, 1], axis=(0, 1))  # Mean of v over time and x

# Reynolds stress (u'v' term)
u_fluct = UV[:, :, :, 0] - U_mean  # Fluctuations in u
v_fluct = UV[:, :, :, 1] - V_mean  # Fluctuations in v
reynolds_stress = np.mean(u_fluct * v_fluct, axis=(0, 1))  # Average of u'v'
U_var = np.mean(u_fluct ** 2, axis=(0, 1))  # Variance of u

# Plot U_mean (Mean of u) vs y
plt.plot(U_mean, y[0])
plt.ylim(y[0][0], y[0][-1])
plt.xlim(0, 1.2)
plt.xlabel('$\overline{u}$', fontsize=18)
plt.ylabel('$y$', fontsize=18)
plt.xticks(fontsize=12)  # Increase tick label font size
plt.yticks(fontsize=12)  # Increase tick label font size
plt.tight_layout()
plt.show()

# Plot U_var (Variance of u) vs y
plt.plot(U_var, y[0])
plt.ylim(y[0][0], y[0][-1])
plt.xlim(0, 0.11)
plt.xlabel(r"$\overline{u'^2}$", fontsize=18)
plt.ylabel('$y$', fontsize=18)
plt.xticks(fontsize=12)  # Increase tick label font size
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# Plot Reynolds stress (u'v') vs y
plt.plot(reynolds_stress, y[0])
plt.ylim(y[0][0], y[0][-1])
# plt.xlim(-0.011, 0.011)
plt.axvline(0, color='black', linestyle='--')
plt.xlabel(r"$\overline{u'v'}$", fontsize=18)
plt.ylabel('$y$', fontsize=18)
plt.xticks(fontsize=12)  # Increase tick label font size
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# Mean v over time and x (so V_mean will be a function of y only)
V_mean = np.mean(UV[:, :, :, 1], axis=(0, 2))  # Mean of v over time and x, shape (148,)

# Reshape V_mean to (1, 148, 1) for broadcasting across time and x-coordinates
V_mean_broadcasted = V_mean[:, np.newaxis]  # Shape: (148, 1)

# Fluctuations in v
v_fluct = UV[:, :, :, 1] - V_mean_broadcasted  # Subtract the mean along y-axis

# Variance of v over time and x (pointwise)
V_var = np.mean(v_fluct ** 2, axis=(0, 2))  # Variance of v, shape (148,)

# Plot V_mean (Mean of v) vs y
plt.plot(x[:, 0], V_mean, color='orange')  # Assuming y[:, 0] corresponds to y-values
plt.xlim(x[0, 0], x[-1, 0])  # Adjust x-axis limits (for y-values)
plt.axhline(0, color='black', linestyle='--')
plt.ylabel(r'$\overline{v}$', fontsize=18)
plt.xlabel('x', fontsize=18)
plt.xticks(fontsize=14)  # Increase tick label font size
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()

# Plot V_var (Variance of v) vs y
plt.plot(x[:, 0], V_var, color='orange')  # Assuming y[:, 0] corresponds to y-values
plt.xlim(x[0, 0], x[-1, 0])  # Adjust x-axis limits (for y-values)
plt.ylim(0, 0.06)
plt.ylabel(r"$\overline{v'^2}$", fontsize=18)
plt.xlabel('x', fontsize=18)
plt.xticks(fontsize=18)  # Increase tick label font size
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()

# reynolds stress
reynolds_stress = np.mean(u_fluct * v_fluct, axis=(0, 2))  # Average of u'v'

# Plot Reynolds stress (u'v') vs y
plt.plot(x[:, 0], reynolds_stress, color='orange')  # Assuming y[:, 0] corresponds to y-values
plt.xlim(x[0, 0], x[-1, 0])  # Adjust x-axis limits (for y-values)
plt.axhline(0, color='black', linestyle='--')
plt.ylabel(r"$\overline{u'v'}$", fontsize=18)
plt.xlabel('x', fontsize=18)
plt.xticks(fontsize=18)  # Increase tick label font size
plt.yticks(fontsize=18)

plt.tight_layout()
plt.show()