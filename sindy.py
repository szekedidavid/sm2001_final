import numpy as np

from helpers import plot_velocities, read_data
from functools import partial
import matplotlib.pyplot as plt
import itertools

def sparsify_dynamics(Theta, Xs_dot, thresh, iter):
    # sparsify dynamics
    Xi = np.linalg.lstsq(Theta, Xs_dot, rcond=None)[0]
    # for k in range(iter):
    #     smallinds = np.abs(Xi) < thresh
    #     Xi[smallinds] = 0
    #     for ind in range(Xs_dot.shape[1]):
    #         biginds = (smallinds[:, ind] == False)
    #         Xi[biginds, ind] = np.linalg.lstsq(Theta[:, biginds], Xs_dot[:, ind], rcond=None)[0]

    return Xi

X, n, m, UV, x, y = read_data()

u_min = np.min(UV[:, :, :, 0])
u_max = np.max(UV[:, :, :, 0])
v_min = np.min(UV[:, :, :, 1])
v_max = np.max(UV[:, :, :, 1])

plot_vel = partial(plot_velocities, x=x, y=y, n=n)
X_norm = np.zeros_like(X)
X_norm[:n // 2] = X[:n // 2] - np.mean(X[:n // 2])
X_norm[n // 2:] = X[n // 2:] - np.mean(X[n // 2:])

svd = np.linalg.svd(X_norm, full_matrices=False)
U = svd[0]
S = np.diag(svd[1])
V = svd[2].T



function_dict = {}

r = 10 # truncation
U_red = U[:, :r]
S_red = S[:r, :r]
V_red = V[:, :r]
Xs = V_red  # time series of each POD mode, time extends downwards
V_red_dot = np.diff(V_red, axis=0)
V_red_dot = np.vstack([V_red_dot, V_red_dot[-1, :]])

def constant_term(td):
    return np.ones(td.shape[0])

def generate_polynomial_term(coef_indices):
    def polynomial_term(td):
        return np.prod([td[:, i] for i in coef_indices], axis=0)
    return polynomial_term

def polynomial_generator(degree):
    for coef_indices in itertools.combinations_with_replacement(range(r), degree):
        yield str(coef_indices), generate_polynomial_term(coef_indices)

function_dict['const'] = constant_term
degree = 1
for i in range(1, degree + 1):
    for poly in polynomial_generator(i):
        function_dict[poly[0]] = poly[1]

Theta = np.zeros((m, len(function_dict)))
for i, func in enumerate(function_dict.values()):
    Theta[:, i] = func(Xs)

iter = 10
thresh = 1e-2
Xi = sparsify_dynamics(Theta, V_red_dot, thresh, iter)
print(Xi)

# reconstruct flow
V_dot_sindy = Theta @ Xi
V_sindy = np.zeros_like(Xs)
V_sindy[0] = Xs[0]
for i in range(1, m):
    V_sindy[i] = V_sindy[i - 1] + V_dot_sindy[i - 1]

X_sindy = U_red @ S_red @ V_sindy.T

# plot reconstructed flow
times = 0, 300, 600, 900
for i in times:
    plot_vel(X_norm[:, i], u_min=u_min, u_max=u_max, v_min=v_min, v_max=v_max)
    plot_vel(X_sindy[:, i], u_min=u_min, u_max=u_max, v_min=v_min, v_max=v_max)

# print RMS
rms = np.sqrt(np.mean((X_norm - X_sindy) ** 2))
print(f'RMS: {rms}')
