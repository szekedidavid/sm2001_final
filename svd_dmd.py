import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

UV = np.load('UV.npy')
x = np.load('x.npy')
y = np.load('y.npy')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
quiver = ax1.quiver(x, y, UV[0, :, :, 0], UV[0, :, :, 1])
ax1.set_title('Velocity vector field')
c_u = ax2.pcolormesh(x, y, UV[0, :, :, 0], cmap='coolwarm', vmin=-1.6, vmax=1.6)
fig.colorbar(c_u, ax=ax2)
ax2.set_title('u component')
c_v = ax3.pcolormesh(x, y, UV[0, :, :, 1], cmap='coolwarm', vmin=-1.6, vmax=1.6)
fig.colorbar(c_v, ax=ax3)
ax3.set_title('v component')


def update(t):
    quiver.set_UVC(UV[t, :, :, 0], UV[t, :, :, 1])
    c_u.set_array(UV[t, :, :, 0].ravel())
    c_v.set_array(UV[t, :, :, 1].ravel())
    return quiver, c_u, c_v


ani = animation.FuncAnimation(fig, update, frames=UV.shape[0], interval=10)
plt.show()
