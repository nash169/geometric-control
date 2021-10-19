#! /usr/bin/env python
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import sys

from matplotlib import cm, colors
from io_utils import get_data

data = get_data(sys.argv[1], "CHART", "EMBEDDING",
                "FUNCTION", "GRAD", "RECORD", "PROJECTION")

chart = data["CHART"]
embedding = data["EMBEDDING"]
function = data["FUNCTION"]
grad = -data["GRAD"]
ds = data["RECORD"]
proj = data["PROJECTION"]

res = int(np.sqrt(len(function)))


# EMBEDDING
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_box_aspect((np.ptp(embedding[:, 0]), np.ptp(
    embedding[:, 1]), np.ptp(embedding[:, 2])))

surf = ax.plot_surface(
    embedding[:, 0].reshape((res, res), order='F'),
    embedding[:, 1].reshape((res, res), order='F'),
    embedding[:, 2].reshape((res, res), order='F'),
    facecolors=cm.jet(function.reshape(
        (res, res), order='F')/np.amax(function)),
    antialiased=False, linewidth=0, alpha=0.2
)
fig.colorbar(surf, ax=ax)

ax.plot(proj[:, 0], proj[:, 1], proj[:, 2], color="black")
ax.scatter(0.0707372, -0.987513,  0.140767)


# CHART
fig = plt.figure()
ax = fig.add_subplot(111)
contour = ax.contourf(
    chart[:, 0].reshape((res, res), order='F'),
    chart[:, 1].reshape((res, res), order='F'),
    function.reshape((res, res), order='F'),
    cmap=cm.jet,
    antialiased=True,
)
# ax.set_aspect("box")
fig.colorbar(contour, ax=ax)

ax.quiver(chart[:, 0].reshape((res, res), order='F'),
          chart[:, 1].reshape((res, res), order='F'),
          grad[:, 0].reshape((res, res), order='F'),
          grad[:, 1].reshape((res, res), order='F'))

ax.plot(ds[:, 1], ds[:, 2], color="black")
ax.scatter(1.5, 3)


# DYNAMICS
fig = plt.figure()
ax = fig.add_subplot(121)
ax.plot(ds[:, 0], ds[:, 1])
ax.plot(ds[:, 0], ds[:, 2])
ax.hlines(1.5, np.min(ds[:, 0]), np.max(ds[:, 0]),
          color="black", linestyles="dashed")
ax.hlines(3, np.min(ds[:, 0]), np.max(ds[:, 0]),
          color="black", linestyles="dashed")

ax = fig.add_subplot(122)
ax.plot(ds[:, 0], ds[:, 3])
ax.plot(ds[:, 0], ds[:, 4])
ax.hlines(0.0, np.min(ds[:, 0]), np.max(ds[:, 0]),
          color="black", linestyles="dashed")


plt.show()
