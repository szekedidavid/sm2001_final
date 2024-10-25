import numpy as np

from helpers import plot_velocities, read_data
from functools import partial
import matplotlib.pyplot as plt
import itertools
import matplotlib

def sparsify_dynamics(Theta, Xs_dot, thresh, iter, tol=1e-5):
    # sparsify dynamics
    Xi = np.linalg.lstsq(Theta, Xs_dot, rcond=None)[0]
    for k in range(iter):
        Xi_old = Xi.copy()
        print('Iteration', k)
        smallinds = np.abs(Xi) < thresh
        Xi[smallinds] = 0
        for ind in range(Xs_dot.shape[1]):
            biginds = (smallinds[:, ind] == False)
            Xi[biginds, ind] = np.linalg.lstsq(Theta[:, biginds], Xs_dot[:, ind], rcond=None)[0]
        if np.linalg.norm(Xi_old - Xi) < tol:
            print('Converged')
            break

    return Xi

X, X_norm, n, m, x, y, UV, u_min, u_max, v_min, v_max = read_data()
plot_vel = partial(plot_velocities, x=x, y=y, n=n)

svd = np.linalg.svd(X_norm, full_matrices=False)
U = svd[0]
S = np.diag(svd[1])
V = svd[2].T

function_dict = {}
poly_terms_counts = {}

r = 10 # truncation
U_red = U[:, :r]
S_red = S[:r, :r]
V_red = V[:, :r]
V_red_dot = np.diff(V_red, axis=0)
V_red_dot = np.vstack([V_red_dot, V_red_dot[-1, :]])

def constant_term(td):
    return np.ones(td.shape[0]) / np.linalg.norm(np.ones(td.shape[0]))

def generate_polynomial_term(coef_indices):
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



function_dict['const'] = constant_term
max_degree = 2
for name, func in polynomial_generator(max_degree):
    function_dict[name] = func


Theta = np.zeros((m, len(function_dict)))
for i, func in enumerate(function_dict.values()):
    Theta[:, i] = func(V_red)

iter = 100
thresh = 0.01
Xi = sparsify_dynamics(Theta, V_red_dot, thresh, iter)

# plot Xi
print(Xi.shape)
fig = plt.figure(figsize=(10, 10))
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
    plt.show()
    plot_vel(X_sindy[:, i], u_min=u_min, u_max=u_max, v_min=v_min, v_max=v_max)
    plt.show()

# print min MSE from POD
X_pod = U_red @ S_red @ V_red.T
mse_pod = np.mean((X_norm - X_pod) ** 2)
print(f'MSE POD: {mse_pod}')

# print MSE
mse = np.mean((X_norm - X_sindy) ** 2)
print(f'MSE SINDy: {mse}')  # todo shift mode and model selection
