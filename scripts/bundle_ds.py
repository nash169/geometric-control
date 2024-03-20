#! /usr/bin/env python
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.linalg import norm
from matplotlib import colors
from io_utils import get_data

data = get_data(sys.argv[1], "EMBEDDING", "POTENTIAL",
                "RECORD", "RADIUS", "CENTER", "TARGET", "EFFECTOR")

res = int(np.sqrt(len(data["POTENTIAL"])))
Xe = data["EMBEDDING"][:, 0].reshape((res, res), order='F')
Ye = data["EMBEDDING"][:, 1].reshape((res, res), order='F')
Ze = data["EMBEDDING"][:, 2].reshape((res, res), order='F')
Fp = data["POTENTIAL"].reshape((res, res), order='F')
Fp -= np.min(Fp)
Fp /= np.max(Fp)

ds = data["RECORD"]
target = data["TARGET"]
radius = data["RADIUS"]
centers = data["CENTER"]
if len(centers.shape) == 1:
    centers = centers[np.newaxis, :]
effector = data["EFFECTOR"]

# COLORMAP
colors = plt.cm.jet(Fp)
mappable = plt.cm.ScalarMappable(cmap=plt.cm.jet)
mappable.set_array(Fp)

# PLOT EMBEDDING
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
surf = ax.plot_surface(Xe, Ye, Ze, facecolors=colors, antialiased=True, linewidth=0, alpha=1.0)
# fig.colorbar(mappable,  ax=ax, label=r"$\phi$")
ax.set_box_aspect((np.ptp(Xe), np.ptp(Ye), np.ptp(Ze)))

# PLOT OBSTACLES
obstacles = data["EMBEDDING"]
for i in range(centers.shape[0]):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = centers[i, 0] + radius*np.cos(u)*np.sin(v)
    y = centers[i, 1] + radius*np.sin(u)*np.sin(v)
    z = centers[i, 2] + radius*np.cos(v)
    ax.plot_surface(x, y, z, linewidth=0.0, cstride=1, rstride=1)
    obstacles = np.append(
        obstacles, np.concatenate((x.flatten()[:, np.newaxis], y.flatten()[:, np.newaxis], z.flatten()[:, np.newaxis]), axis=1), axis=0)
obstacles = np.append(obstacles, ds[:, 1:4], axis=0)
ax.set_box_aspect((np.ptp(obstacles[:, 0]), np.ptp(obstacles[:, 1]), np.ptp(obstacles[:, 2])))
ax.view_init(elev=-150, azim=60)

# PLOT TRAJECTORY
check = norm((ds[:, 1:4] - np.array([[0.7, 0.0, 0.5]]))/0.3, axis=1) < 1
ax.plot(ds[:, 1], ds[:, 2], ds[:, 3], color="black", label="Trajectory", zorder=20)
ax.plot(ds[check, 1], ds[check, 2], ds[check, 3], color="red")
# target
ax.scatter(target[0], target[1],  target[2], color="red", label="Target")
# init pos
# ax.scatter(ds[0, 1], ds[0, 2], ds[0, 3], color="green", label="Initial Position", zorder=20)
# end pos
ax.scatter(ds[-1, 1], ds[-1, 2], ds[-1, 3], s=200, edgecolors='k', c='yellow', marker="*", label="Final Position", zorder=20)
# init vel
# ax.quiver(ds[0, 1], ds[0, 2], ds[0, 3], ds[0, 4], ds[0, 5], ds[0, 6], length=50, color='k')
# ax.set_title('Sampled trajectory on the manifold')
# ax.legend(loc="lower left")

ax.axis('off')
fig.patch.set_visible(False)
fig.tight_layout()
fig.savefig('intro_plane_ds_50.png', format='png', dpi=100, bbox_inches="tight")

# # DYNAMICS
# fig = plt.figure()
# ax = fig.add_subplot(121)
# ax.plot(ds[:, 0], ds[:, 1], color="C0", label="X position")
# ax.plot(ds[:, 0], ds[:, 2], color="C1", label="Y position")
# ax.plot(ds[:, 0], ds[:, 3], color="C2", label="Z position")
# if effector.size != 0:
#     ax.plot(ds[:, 0], effector[:, 0], color="C0", linestyle="dashed")
#     ax.plot(ds[:, 0], effector[:, 1], color="C1", linestyle="dashed")
#     ax.plot(ds[:, 0], effector[:, 2], color="C2", linestyle="dashed")
# ax.hlines(target[0], np.min(ds[:, 0]), np.max(ds[:, 0]),
#           color="black", linestyles="dashed")
# ax.hlines(target[1], np.min(ds[:, 0]), np.max(ds[:, 0]),
#           color="black", linestyles="dashed")
# ax.hlines(target[2], np.min(ds[:, 0]), np.max(ds[:, 0]),
#           color="black", linestyles="dashed")
# ax.legend(loc="upper right")
# ax.set_xlabel('Time [s]')
# ax.set_ylabel('Position [m]')
# ax.set_title('Position Profiles')

# ax = fig.add_subplot(122)
# ax.plot(ds[:, 0], ds[:, 4], label="X velocity")
# ax.plot(ds[:, 0], ds[:, 5], label="Y velocity")
# ax.plot(ds[:, 0], ds[:, 6], label="Z velocity")
# ax.hlines(0.0, np.min(ds[:, 0]), np.max(ds[:, 0]),
#           color="black", linestyles="dashed", label="Velocity target")
# ax.legend(loc="upper right")
# ax.set_xlabel('Time [s]')
# ax.set_ylabel('Velocities [m/s]')
# ax.set_title('Velocity Profiles')


plt.show()
