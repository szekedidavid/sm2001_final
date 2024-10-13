import numpy as np
import jax.numpy as jnp
import jax

from helpers import plot_velocities
from functools import partial

print(jax.devices())


@jax.jit
def shrink(X, tau):
    Y = jnp.abs(X) - tau
    return jnp.sign(X) * jnp.maximum(Y, jnp.zeros_like(Y))


@jax.jit
def SVT(X, tau):
    # U, S, VT = np.linalg.svd(X, full_matrices=False)
    # out = U @ np.diag(shrink(S, tau)) @ VT
    # return out
    U, S, VT = jnp.linalg.svd(X, full_matrices=False)
    out = U @ jnp.diag(shrink(S, tau)) @ VT
    return out


@jax.jit
def iterate_RPCA(X, S, L, Y, mu, lambd, thresh):
    L = SVT(X - S + (1 / mu) * Y, 1 / mu)
    S = shrink(X - L + (1 / mu) * Y, lambd / mu)
    Y = Y + mu * (X - L - S)
    return L, S, Y


def RPCA(X):
    n1, n2 = X.shape
    mu = n1 * n2 / (4 * np.sum(np.abs(X.reshape(-1))))
    lambd = 1 / np.sqrt(np.maximum(n1, n2))
    thresh = 10 ** (-7) * np.linalg.norm(X)

    X = jnp.array(X)
    S = jnp.zeros_like(X)
    Y = jnp.zeros_like(X)
    L = jnp.zeros_like(X)

    count = 0
    while (jnp.linalg.norm(X - L - S) > thresh) and (count < 1000):
        print("Norm diff: ", jnp.linalg.norm(X - L - S), "Threshold: ", thresh, "Iteration: ", count)
        L, S, Y = iterate_RPCA(X, S, L, Y, mu, lambd, thresh)
        count += 1

    return L, S


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

# normalize X
X_norm = np.zeros_like(X)
X_norm[:n // 2] = X[:n // 2] - np.mean(X[:n // 2])
X_norm[n // 2:] = X[n // 2:] - np.mean(X[n // 2:])

L, S = RPCA(X_norm)
# save L
np.save('data/L.npy', L)
# save S
np.save('data/S.npy', S)
