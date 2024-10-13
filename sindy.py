import numpy as np

from helpers import plot_velocities, read_data
from functools import partial
import matplotlib.pyplot as plt
import itertools
import matplotlib

def sparsify_dynamics(Theta, Xs_dot, thresh, iter):
    # sparsify dynamics
    Xi = np.linalg.lstsq(Theta, Xs_dot, rcond=None)[0]
    for k in range(iter):
        print('Iteration', k)
        smallinds = np.abs(Xi) < thresh
        Xi[smallinds] = 0
        for ind in range(Xs_dot.shape[1]):
            biginds = (smallinds[:, ind] == False)
            Xi[biginds, ind] = np.linalg.lstsq(Theta[:, biginds], Xs_dot[:, ind], rcond=None)[0]

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
V_red_dot = np.diff(V_red, axis=0)
V_red_dot = np.vstack([V_red_dot, V_red_dot[-1, :]])

# get lower limit for reconstruction RMS
X_svd = U_red @ S_red @ V_red.T
rms = np.sqrt(np.mean((X_norm - X_svd) ** 2))
print(f'Lower limit RMS: {rms}')

def constant_term(td):
    return np.ones(td.shape[0]) / np.linalg.norm(np.ones(td.shape[0]))

poly_terms_counts = {}

def generate_polynomial_term(coef_indices):  # todo enforce that average flow is constant
    def polynomial_term(td):
        product = np.prod([td[:, i] for i in coef_indices], axis=0)
        normalized_product = product / np.linalg.norm(product)
        return normalized_product
    return polynomial_term

def polynomial_generator(max_degree):
    for degree in range(1, max_degree + 1):
        poly_terms_counts[degree] = 0
        for coef_indices in itertools.combinations_with_replacement(range(r), degree):
            poly_terms_counts[degree] += 1
            yield f'poly_{coef_indices}', generate_polynomial_term(coef_indices)

# def generate_sin_term(coef_index, freq):
#     def sin_term(td):
#         return np.sin(freq * td[:, coef_index])
#     return sin_term
#
# def generate_cos_term(coef_index, freq):
#     def cos_term(td):
#         return np.cos(freq * td[:, coef_index])
#     return cos_term
#
# # generate sin and cos terms
# def trig_generator(max_freq):
#     for freq in range(1, max_freq + 1):
#         for i in range(r):
#             yield f'sin_{freq}_{i}', generate_sin_term(i, freq)
#             yield f'cos_{freq}_{i}', generate_cos_term(i, freq)

function_dict['const'] = constant_term
max_degree = 3
for name, func in polynomial_generator(max_degree):
    function_dict[name] = func


Theta = np.zeros((m, len(function_dict)))
for i, func in enumerate(function_dict.values()):
    Theta[:, i] = func(V_red)

iter = 10
thresh = 1e-2  # todo higher modes generally need more terms?
Xi = sparsify_dynamics(Theta, V_red_dot, thresh, iter)
# plot Xi
# aspect ratio is square
fig = plt.figure(figsize=(10, 10))
# plt.matshow(Xi, cmap='coolwarm', aspect='equal')
# logartihmic color scale
ax = fig.add_subplot(111)
cax = ax.matshow(Xi, cmap='coolwarm', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=thresh))
count_pos = 0.5
for count in poly_terms_counts.values():
    count_pos += count
    ax.axhline(count_pos, color='black')
fig.colorbar(cax)
plt.show()

# reconstruct flow
V_dot_sindy = Theta @ Xi
V_sindy = np.zeros_like(V_red)
V_sindy[0] = V_red[0]
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
print(f'RMS: {rms}')  # todo shift mode and model selection
