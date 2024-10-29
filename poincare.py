import numpy as np
import matplotlib.pyplot as plt
# import kde from scipy
from scipy.stats import gaussian_kde


def poincare_map(V, V_pred, index_1, index_2, index_3):
    indices = np.where(np.diff(np.sign(V[:, index_2])) == -2)[0]

    poincare_points_x = V[indices, index_1]
    poincare_points_y = V[indices, index_3]

    indices_pred = np.where(np.diff(np.sign(V_pred[:, index_2])) == -2)[0]

    poincare_points_x_pred = V_pred[indices_pred, index_1]
    poincare_points_y_pred = V_pred[indices_pred, index_3]

    # calculate KDE
    kde = gaussian_kde([poincare_points_x, poincare_points_y])
    kde_pred = gaussian_kde([poincare_points_x_pred, poincare_points_y_pred])

    plt.figure(figsize=(8, 6))
    plt.scatter(poincare_points_x_pred, poincare_points_y_pred, color='red', s=10, label='Predicted')
    plt.scatter(poincare_points_x, poincare_points_y, color='blue', s=10, label='Real')

    plt.grid(True)
    x = np.linspace(np.min(poincare_points_x), np.max(poincare_points_x), 100)
    y = np.linspace(np.min(poincare_points_y), np.max(poincare_points_y), 100)
    X, Y = np.meshgrid(x, y)
    Z = kde([X.ravel(), Y.ravel()])
    Z = Z.reshape(X.shape)
    # plt.contour(X, Y, Z, colors='blue', alpha=0.5)
    x_pred = np.linspace(np.min(poincare_points_x_pred), np.max(poincare_points_x_pred), 100)
    y_pred = np.linspace(np.min(poincare_points_y_pred), np.max(poincare_points_y_pred), 100)
    X_pred, Y_pred = np.meshgrid(x_pred, y_pred)
    Z_pred = kde_pred([X_pred.ravel(), Y_pred.ravel()])
    Z_pred = Z_pred.reshape(X_pred.shape)
    # plt.contour(X_pred, Y_pred, Z_pred, colors='red', alpha=0.5)
    plt.xlim(-0.08, 0.08)
    plt.ylim(-0.08, 0.08)
    plt.xlabel(f'$v_{{{index_1}}}$', fontsize=18)
    plt.ylabel(f'$v_{{{index_3}}}$', fontsize=18)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()


# load V_red and V_pred from the file
V_red = np.load('V_red.npy')[:400]
V_pred = np.load('V_pred.npy')[:400]

# plot the Poincaré map
poincare_map(V_red, V_pred, 0, 1, 12)
