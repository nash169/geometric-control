#! /usr/bin/env python
# encoding: utf-8

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from scipy.linalg import norm
from matplotlib import colors
from io_utils import get_data

data = get_data("outputs/robot_bundle.csv",
                "TIME", "CONFIG", "TASK",
                "TARGET", "RADIUS", "CENTER",
                "EMBEDDING", "POTENTIAL",
                "POSLIMITS", "VELLIMITS")

res = int(np.sqrt(len(data["POTENTIAL"])))
Xe = data["EMBEDDING"][:, 0].reshape((res, res), order='F')
Ye = data["EMBEDDING"][:, 1].reshape((res, res), order='F')
Ze = data["EMBEDDING"][:, 2].reshape((res, res), order='F')
Fp = data["POTENTIAL"].reshape((res, res), order='F')
Fp -= np.min(Fp)
Fp /= np.max(Fp)

time = data["TIME"]
robot_pos = data["CONFIG"][:, :7]
robot_vel = data["CONFIG"][:, 7:]
fk_pos = data["TASK"][:, :3]
fk_vel = data["TASK"][:, 3:6]
task_pos = data["TASK"][:, 6:-3]
task_vel = data["TASK"][:, -3:]
pos_limits = data["POSLIMITS"]
vel_limits = data["VELLIMITS"]

target = data["TARGET"]
radius = data["RADIUS"]
centers = data["CENTER"]
if len(centers.shape) == 1:
    centers = centers[np.newaxis, :]

# COLORMAP
colors = plt.cm.jet(Fp)
mappable = plt.cm.ScalarMappable(cmap=plt.cm.jet)
mappable.set_array(Fp)

# PLOT EMBEDDING
fig_1 = plt.figure()
ax = fig_1.add_subplot(111, projection="3d")
surf = ax.plot_surface(Xe, Ye, Ze, facecolors=colors,
                       antialiased=True, linewidth=0, alpha=0.5)
fig_1.colorbar(mappable,  ax=ax, label=r"$\phi$")
# ax.set_box_aspect((np.ptp(Xe), np.ptp(Ye), np.ptp(Ze)))

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
obstacles = np.append(obstacles, fk_pos, axis=0)
ax.set_box_aspect((np.ptp(obstacles[:, 0]), np.ptp(
    obstacles[:, 1]), np.ptp(obstacles[:, 2])))
ax.view_init(20, 160)

# PLOT TRAJECTORY
ax.plot(fk_pos[:, 0], fk_pos[:, 1], fk_pos[:, 2],
        color="black", label="Robot FK")
ax.plot(task_pos[:, 0], task_pos[:, 1], task_pos[:, 2],
        color="black", linestyle="dashed", label="Task DS")
# target
ax.scatter(target[0], target[1],  target[2], color="red", label="Target")
# init pos
ax.scatter(fk_pos[0, 0], fk_pos[0, 1], fk_pos[0, 2],
           color="green", label="Initial Position")
ax.scatter(task_pos[0, 0], task_pos[0, 1], task_pos[0, 2], color="green")
# end pos
ax.scatter(fk_pos[-1, 0], fk_pos[-1, 1], fk_pos[-1, 2],
           color="blue", label="Final Position")
ax.scatter(task_pos[-1, 0], task_pos[-1, 1], task_pos[-1, 2], color="blue")
# init vel
ax.quiver(fk_pos[0, 0], fk_pos[0, 2], fk_pos[0, 2], fk_vel[0, 0],
          fk_vel[0, 1], fk_vel[0, 2], length=50, color='k')
ax.quiver(task_pos[0, 0], task_pos[0, 1], task_pos[0, 2], task_vel[0, 0],
          task_vel[0, 1], task_vel[0, 2], length=50, color='k')
ax.set_title('Sampled trajectory on the manifold')
ax.legend(loc="lower left")


# DYNAMICS
fig = plt.figure()
ax = fig.add_subplot(121)
ax.plot(time, fk_pos[:, 0], color="C0", label="X position")
ax.plot(time, fk_pos[:, 1], color="C1", label="Y position")
ax.plot(time, fk_pos[:, 2], color="C2", label="Z position")

ax.plot(time, task_pos[:, 0], color="C0", linestyle="dashed")
ax.plot(time, task_pos[:, 1], color="C1", linestyle="dashed")
ax.plot(time, task_pos[:, 2], color="C2", linestyle="dashed")

ax.hlines(target[0], np.min(time), np.max(time),
          color="black", linestyles="dashed")
ax.hlines(target[1], np.min(time), np.max(time),
          color="black", linestyles="dashed")
ax.hlines(target[2], np.min(time), np.max(time),
          color="black", linestyles="dashed")

ax.legend(loc="upper right")
ax.set_xlabel('Time [s]')
ax.set_ylabel('Position [m]')
ax.set_title('Position Profiles')

ax = fig.add_subplot(122)
ax.plot(time, fk_vel[:, 0], color="C0", label="X velocity")
ax.plot(time, fk_vel[:, 1], color="C1", label="Y velocity")
ax.plot(time, fk_vel[:, 2], color="C2", label="Z velocity")

ax.plot(time, task_vel[:, 0], color="C0", linestyle="dashed")
ax.plot(time, task_vel[:, 1], color="C1", linestyle="dashed")
ax.plot(time, task_vel[:, 2], color="C2", linestyle="dashed")

ax.hlines(0.0, np.min(time), np.max(time),
          color="black", linestyles="dashed", label="Velocity target")
ax.legend(loc="upper right")
ax.set_xlabel('Time [s]')
ax.set_ylabel('Velocities [m/s]')
ax.set_title('Velocity Profiles')

# PHASE CONFIGURATION SPACE MOTION
fig = plt.figure(figsize=(25, 3))
for i in np.arange(7):
    if i < 6:
        ax = fig.add_subplot(1, 6, i+1)
        ax.plot(robot_pos[:, i], robot_vel[:, i],
                color="C" + str(i), label="Joint " + str(i))
        # ax.vlines(pos_limits[i, 0], np.min(robot_pos[:, i]), np.max(robot_pos[:, i]),
        #           color="black", linestyles="dashed")
        # ax.vlines(pos_limits[i, 1], np.min(robot_pos[:, i]), np.max(robot_pos[:, i]),
        #           color="black", linestyles="dashed")
        # ax.hlines(vel_limits[i, 0], np.min(robot_vel[:, i]), np.max(robot_vel[:, i]),
        #           color="black", linestyles="dashed")
        # ax.hlines(vel_limits[i, 1], np.min(robot_vel[:, i]), np.max(robot_vel[:, i]),
        #           color="black", linestyles="dashed")
        ax.set_xlabel('q')
        if i == 0:
            ax.set_ylabel('q_dot')
# ax.legend(loc="upper right")
# ax.set_title('Configuration Phase Space')

plt.show()
record = False

if record:
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    # Video trajectories on the sphere
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Xe, Ye, Ze, facecolors=colors,
                    antialiased=True, linewidth=0, alpha=0.5)
    fig.colorbar(mappable,  ax=ax, label=r"$\phi$")
    for i in range(centers.shape[0]):
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = centers[i, 0] + radius*np.cos(u)*np.sin(v)
        y = centers[i, 1] + radius*np.sin(u)*np.sin(v)
        z = centers[i, 2] + radius*np.cos(v)
        ax.plot_surface(x, y, z, linewidth=0.0, cstride=1, rstride=1)
    ax.set_box_aspect((np.ptp(obstacles[:, 0]), np.ptp(
        obstacles[:, 1]), np.ptp(obstacles[:, 2])))
    # target
    ax.scatter(target[0], target[1],  target[2], color="red", label="Target")
    # init pos
    ax.scatter(fk_pos[0, 0], fk_pos[0, 1], fk_pos[0, 2],
               color="green", label="Initial Position")
    ax.scatter(task_pos[0, 0], task_pos[0, 1], task_pos[0, 2], color="green")
    ax.set_title('Sampled trajectory on the manifold')
    ax.legend(loc="lower left")
    ax.view_init(20, 160)

    with writer.saving(fig, "outputs/video/sphere_traj.mp4", 200):
        for i in np.arange(0, len(time), 1000):
            ax.plot(fk_pos[:i, 0], fk_pos[:i, 1], fk_pos[:i, 2],
                    color="black", label="Robot FK")
            ax.plot(task_pos[:i, 0], task_pos[:i, 1], task_pos[:i, 2], color="black",
                    linestyle="dashed", label="Task DS")
            # ax.scatter(fk_pos[i, 0], fk_pos[i, 1], fk_pos[i, 2],
            #            color="blue", label="FK Final Position")
            # ax.scatter(task_pos[i, 0], task_pos[i, 1], task_pos[i, 2],
            #            color="blue", label="Task Final Position")
            print(i)
            writer.grab_frame()

    codeBASH = "ffmpeg -i outputs/video/sphere_traj.mp4 -filter_complex 'fps=10,scale=640:-1:flags=lanczos,split [o1] [o2];[o1] palettegen [p]; [o2] fifo [o3];[o3] [p] paletteuse' outputs/video/sphere_traj.gif"
    os.system(codeBASH)

    # Video temporal response of the system
    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    ax1.hlines(target[0], np.min(time), np.max(time),
               color="black", linestyles="dashed")
    ax1.hlines(target[1], np.min(time), np.max(time),
               color="black", linestyles="dashed")
    ax1.hlines(target[2], np.min(time), np.max(time),
               color="black", linestyles="dashed")
    ax1.legend(loc="upper right")
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Position [m]')
    ax1.set_title('Position Profiles')

    ax2 = fig.add_subplot(122)
    ax2.hlines(0.0, np.min(time), np.max(time),
               color="black", linestyles="dashed", label="Velocity target")
    ax2.legend(loc="upper right")
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Velocities [m/s]')
    ax2.set_title('Velocity Profiles')

    with writer.saving(fig, "outputs/video/time_response.mp4", 200):
        for i in np.arange(0, len(time), 1000):
            ax1.plot(time[:i], fk_pos[:i, 0], color="C0", label="X position")
            ax1.plot(time[:i], fk_pos[:i, 1], color="C1", label="Y position")
            ax1.plot(time[:i], fk_pos[:i, 2], color="C2", label="Z position")

            ax1.plot(time[:i], task_pos[:i, 0], color="C0", linestyle="dashed")
            ax1.plot(time[:i], task_pos[:i, 1], color="C1", linestyle="dashed")
            ax1.plot(time[:i], task_pos[:i, 2], color="C2", linestyle="dashed")

            ax2.plot(time[:i], fk_vel[:i, 0], color="C0", label="X velocity")
            ax2.plot(time[:i], fk_vel[:i, 1], color="C1", label="Y velocity")
            ax2.plot(time[:i], fk_vel[:i, 2], color="C2", label="Z velocity")

            ax2.plot(time[:i], task_vel[:i, 0], color="C0", linestyle="dashed")
            ax2.plot(time[:i], task_vel[:i, 1], color="C1", linestyle="dashed")
            ax2.plot(time[:i], task_vel[:i, 2], color="C2", linestyle="dashed")

            print(i)
            writer.grab_frame()

    codeBASH = "ffmpeg -i outputs/video/time_response.mp4 -filter_complex 'fps=10,scale=640:-1:flags=lanczos,split [o1] [o2];[o1] palettegen [p]; [o2] fifo [o3];[o3] [p] paletteuse' outputs/video/time_response.gif"
    os.system(codeBASH)

    # Video joint motion
    fig = plt.figure(figsize=(25, 3))
    ax = []
    for i in np.arange(7):
        if i < 6:
            ax.append(fig.add_subplot(1, 6, i+1))
            ax[i].set_xlim(np.min(robot_pos[:, i]), np.max(robot_pos[:, i]))
            ax[i].set_ylim(np.min(robot_vel[:, i]), np.max(robot_vel[:, i]))
            ax[i].set_xlabel('q')
            if i == 0:
                ax[i].set_ylabel('q_dot')

    with writer.saving(fig, "outputs/video/phase_space.mp4", 200):
        for i in np.arange(0, len(time), 1000):
            for j in np.arange(7):
                if j < 6:
                    ax[j].plot(robot_pos[:i, j], robot_vel[:i, j],
                               color="C" + str(j), label="Joint " + str(j))
            print(i)
            writer.grab_frame()

    codeBASH = "ffmpeg -i outputs/video/phase_space.mp4 -filter_complex 'fps=10,scale=640:-1:flags=lanczos,split [o1] [o2];[o1] palettegen [p]; [o2] fifo [o3];[o3] [p] paletteuse' outputs/video/phase_space.gif"
    os.system(codeBASH)
