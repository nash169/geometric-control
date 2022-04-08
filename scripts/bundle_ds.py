#! /usr/bin/env python
# encoding: utf-8

from math import radians
from shutil import which
from termios import VT0
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.linalg import norm
from scipy.linalg import eig
from scipy.sparse.linalg import eigs
from matplotlib import cm, colors
from io_utils import get_data

data = get_data(sys.argv[1], "EMBEDDING", "POTENTIAL",
                "RECORD", "RADIUS", "CENTER")

embedding = data["EMBEDDING"]
function = data["POTENTIAL"]
ds = data["RECORD"]
radius = data["RADIUS"]
centers = data["CENTER"]
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
    antialiased=False, linewidth=0, alpha=0.25
)
fig.colorbar(surf, ax=ax)

# trajectory
ax.plot(ds[:, 1], ds[:, 2], ds[:, 3], color="black")
# target
ax.scatter(0.0707372, -0.987513,  0.140767, color="red")
# init pos
ax.scatter(ds[0, 1], ds[0, 2], ds[0, 3], color="black")
# end pos
ax.scatter(ds[-1, 1], ds[-1, 2], ds[-1, 3], color="blue")
# init vel
ax.quiver(ds[0, 1], ds[0, 2], ds[0, 3], ds[0, 4],
          ds[0, 5], ds[0, 6], length=100, color='k')


# OBSTAClE
for i in range(centers.shape[0]):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = centers[i, 0] + radius*np.cos(u)*np.sin(v)
    y = centers[i, 1] + radius*np.sin(u)*np.sin(v)
    z = centers[i, 2] + radius*np.cos(v)
    ax.plot_surface(x, y, z, linewidth=0.0, cstride=1, rstride=1)


# DYNAMICS
fig = plt.figure()
ax = fig.add_subplot(121)
ax.plot(ds[:, 0], ds[:, 1])
ax.plot(ds[:, 0], ds[:, 2])
ax.plot(ds[:, 0], ds[:, 3])
ax.hlines(0.0707372, np.min(ds[:, 0]), np.max(ds[:, 0]),
          color="black", linestyles="dashed")
ax.hlines(-0.987513, np.min(ds[:, 0]), np.max(ds[:, 0]),
          color="black", linestyles="dashed")
ax.hlines(0.140767, np.min(ds[:, 0]), np.max(ds[:, 0]),
          color="black", linestyles="dashed")

ax = fig.add_subplot(122)
ax.plot(ds[:, 0], ds[:, 4])
ax.plot(ds[:, 0], ds[:, 5])
ax.plot(ds[:, 0], ds[:, 6])
ax.hlines(0.0, np.min(ds[:, 0]), np.max(ds[:, 0]),
          color="black", linestyles="dashed")


plt.show()
