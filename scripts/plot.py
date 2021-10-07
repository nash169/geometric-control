#! /usr/bin/env python
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import sys

from matplotlib import cm
from io_utils import get_data

data = get_data(sys.argv[1], "SAMPLES")

res = 100

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(
    data["SAMPLES"][:, 0].reshape((res, res), order='F'),
    data["SAMPLES"][:, 1].reshape((res, res), order='F'),
    data["SAMPLES"][:, 2].reshape((res, res), order='F'),
    cmap=cm.jet,
    # facecolors=cm.jet(data["FUNCTION"].reshape(
    #     (res, res), order='F')/np.amax(data["FUNCTION"])),
    antialiased=True, linewidth=0
)
fig.colorbar(surf, ax=ax)

fig = plt.figure()
ax = fig.add_subplot(111)
contour = ax.contourf(
    data["SAMPLES"][:, 0].reshape((res, res), order='F'),
    data["SAMPLES"][:, 1].reshape((res, res), order='F'),
    data["SAMPLES"][:, 2].reshape((res, res), order='F'),
    cmap=cm.jet,
    antialiased=True,
)
ax.set_aspect("equal", "box")
fig.colorbar(contour, ax=ax)

plt.show()
