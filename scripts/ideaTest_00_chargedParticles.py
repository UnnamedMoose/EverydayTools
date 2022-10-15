# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 09:09:08 2022

@author: ALidtke
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import os
import re
import platform

font = {"family": "serif",
        "weight": "normal",
        "size": 16}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (9, 6)

from everydaytools import nicePlots

# x, y, type (0=wall, 1=internal), velocityMagnitude
iPos = [0, 1]  # 2D
iPos = [0, 1, 2]  # 3D
particles = np.array([
    # [0., 0., 0, 0.],  # tests 0 and 1
    # [0.25, 0., 0, 0.],
    # [1., 0., 0, 0.],
    # [1., 0.5, 0, 0.],
    # [1., 1., 0, 0.],
    # [0.5, 1., 0, 0.],
    # [0., 0.25, 0, 0.],
    # [0., 1., 0, 0.],
    # [0.1, 0.2, 1, 0.],
    # [0.7, 0.2, 1, 0.],
    # [0.1, 0.6, 1, 0.],
    # [0.8, 0.7, 1, 0.],
    [0., 0., 0., 0.],  # test 2
    [1, 0., 0., 0.],
    [2., 0., 0., 0.],
    [3., 0., 0., 0.],
    [4., 2., 0., 0.],
    [3., 3., 0., 0.],
    [0., 3., 0., 0.],
    [-1., 2., 0., 0.],
    [-0.5, 1., 1., 0.],
    [1.5, 0.1, 1., 0.],
    [1., 3., 1., 0.],
    [-1., 2.2, 1., 0.],
])

edges = [
    # [1, 8],  # test 0 - one particle in between
    # [3, 8],
    # [5, 8],
    # [6, 8],
    # [0, 8],  # test 1 - block in the middle
    # [8, 10],
    # [7, 10],
    # [9, 8],
    # [2, 9],
    # [9, 11],
    # [11, 4],
    # [10, 11],
    [0, 8],
    [7, 8],
    [6, 8],
    [8, 9],
    [1, 9],
    [9, 10],
    [8, 11],
    [2, 10],
    [10, 11],
    [3, 11],
    [4, 11],
    [5, 11],
]

ke = 0
ks = 1
kd = 10
nTimesteps = 2500
dt = 1e-1
velMagFraction = 0.1
velMax = 1
dMin = 1.  # TODO this doesn't work
# When change of total energy becomes less than this, terminate.
tol = 1e-3

# %% Solve.

histories = [particles.copy()]

# Prepare damping that varies in time to help achieve steady state at the end.
def kd(i):
    return (np.tanh(10*(i/nTimesteps)-5) + 1) / 2.

# compute total initial energy - Ep = 0.5 k extension^2
energy = 0.
for iParticle in range(particles.shape[0]):
    relPos = particles[iParticle, :2] - particles[:, :2]
    for iEdge in range(len(edges)):
        if iParticle in edges[iEdge]:
            if edges[iEdge][0] == iParticle:
                iLink = 1
            else:
                iLink = 0
            energy += np.linalg.norm(relPos[edges[iEdge][iLink], :])**2
energy = [energy]

# Iterate in pseudo-time until everything settles down.
for iTime in range(nTimesteps):
    force = np.zeros((particles.shape[0], 2))
    force_spring = np.zeros((particles.shape[0], 2))
    force_damp = np.zeros((particles.shape[0], 2))
    energy = np.append(energy, 0)
    for iParticle in range(particles.shape[0]):
        # Compute relative position.
        relPos = particles[iParticle, :2] - particles[:, :2]
        # Set max velocity based on minimum distance to other particles.
        mask = ~np.isin(np.arange(particles.shape[0]), iParticle)
        particles[iParticle, 3] = min(velMax, velMagFraction*np.min(np.linalg.norm(relPos[mask, :], axis=1)))
        # Set repulsive force.
        force[iParticle, :] = np.sum(
            1*1*1/np.clip(relPos[mask, :]**2, 1e-24, np.inf)*np.sign(relPos[mask, :]), axis=0)
        # Set spring force.
        for iEdge in range(len(edges)):
            if iParticle in edges[iEdge]:
                if edges[iEdge][0] == iParticle:
                    iLink = 1
                else:
                    iLink = 0
                # TODO repulsion doesn't work.
                force_spring[iParticle, :] += -1*relPos[edges[iEdge][iLink], :]# \
                    # * np.sign(np.linalg.norm(relPos[edges[iEdge][iLink], :]) - dMin)
                energy[-1] += np.linalg.norm(relPos[edges[iEdge][iLink], :])**2
                # Set damping force.
                if iTime > 2:
                    force_damp[iParticle, :] += -1.*(particles[iParticle, :2] - histories[-2][iParticle, :2])
    # Non-dimensionalise.
    # TODO for now at least
    force = ke*force/np.clip(np.linalg.norm(force, axis=1)[:, np.newaxis], 1e-24, np.inf)
    force_spring = ks*force_spring/np.clip(np.linalg.norm(force_spring, axis=1)[:, np.newaxis], 1e-24, np.inf)
    force_damp = kd(iTime)*force_damp/np.clip(np.linalg.norm(force_damp, axis=1)[:, np.newaxis], 1e-24, np.inf)

    # Advance.
    mask = particles[:, 2] == 1
    particles[mask, :2] += \
        (force[mask, :]+force_spring[mask, :]+force_damp[mask, :])*particles[mask, 3][:, np.newaxis]*dt
    # Store
    histories.append(particles.copy())

    # check if reduction is small enough in order to end.
    if iTime > 10 and -(energy[-1] - energy[-2]) < tol:
        break

histories = np.array(histories)

# %% Visualise results.

# Check how damping varies in time.
fig, ax = nicePlots.niceFig([], [], "Iteration", "Damping constant")
x = np.arange(nTimesteps)
y = [kd(i) for i in x]
ax.plot(x, y, "r-", lw=2)

# Plot time history of total energy.
fig, ax = nicePlots.niceFig([], [], "Iteration", "Total energy")
ax.plot(energy, "r-", lw=2)

# Plot final grid, forces and free vertex trajectories.
fig, ax = nicePlots.niceFig([], [], "x", "y")
for i in range(particles.shape[0]):
    ax.text(particles[i, 0], particles[i, 1], "{:d}".format(i))
    fScale = 0.2
    ax.plot([particles[i, 0], particles[i, 0]+force[i, 0]*fScale],
            [particles[i, 1], particles[i, 1]+force[i, 1]*fScale], "g-")
    ax.plot([particles[i, 0], particles[i, 0]+force_spring[i, 0]*fScale],
            [particles[i, 1], particles[i, 1]+force_spring[i, 1]*fScale], "m-")
    ax.plot([particles[i, 0], particles[i, 0]+force_damp[i, 0]*fScale],
            [particles[i, 1], particles[i, 1]+force_damp[i, 1]*fScale], "c-")
ax.plot(histories[0, :, 0], histories[0, :, 1], "ks")
ax.plot(histories[-1, :, 0], histories[-1, :, 1], "ro")
for iParticle in range(particles.shape[0]):
    if particles[iParticle, 2] > 0:
        ax.plot(histories[:, iParticle, 0], histories[:, iParticle, 1], "r-")
for edge in edges:
    ax.plot(histories[0, edge, 0], histories[0, edge, 1], "b-", lw=4, alpha=0.25)
    ax.plot(histories[-1, edge, 0], histories[-1, edge, 1], "k-", lw=4, alpha=0.25)

# Plot convergence of position of each mobile vertex.
fig, ax1, ax2 = nicePlots.niceFig([], [], "Iteration", "x", returnTwinAxes=True, twinYlabel="y",
                                  marginL=0.11, marginR=0.89)
for iParticle in range(particles.shape[0]):
    if particles[iParticle, 2] > 0:
        ln = ax1.plot(histories[:, iParticle, 0], label="x")[0]
        ln2 = ax2.plot(histories[:, iParticle, 1], ls="--", label="y", c=ln.get_color())
lns = [ln]+ln2
fig.legend(lns, [l.get_label() for l in lns], ncol=2, loc="upper center")

# Plot only the final grid.
fig, ax = nicePlots.niceFig([], [], "x", "y")
for edge in edges:
    ax.plot(histories[-1, edge, 0], histories[-1, edge, 1], "k-", lw=4, alpha=0.25)
ax.plot(histories[-1, :, 0], histories[-1, :, 1], "ro")
