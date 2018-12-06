## Particle filter implementation for robot localization.
# code adapted from https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
##     RLabbe - Kalman and Bayesian Filters in Python - GitHub

import numpy as np 
from numpy.random import uniform
from filterpy.monte_carlo import systematic_resample
from numpy.linalg import norm
from numpy.random import randn
import scipy.stats
import math


def create_uniform_particles(x_range, y_range, hdg_range, N):
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=N)
    particles[:, 2] %= 2 * np.pi
    return particles

def create_gaussian_particles(mean, std, N):
    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + (randn(N) * std[0])
    particles[:, 1] = mean[1] + (randn(N) * std[1])
    particles[:, 2] = mean[2] + (randn(N) * std[2])
    particles[:, 2] %= 2 * np.pi
    return particles

def predict(particles, u, std, dt=1.):
    N = len(particles)
    
    v = 0.5 * 80* 0.5 * (u[0] + u[1])
    l_dis = dt * u[0]
    r_dis = dt * u[1]
	#update = np.array((sy.cos(theta)*v*rr, sy.sin(theta)*v*rr, (r_dis-l_dis)/1.5))
    update = np.array((v*dt, v*dt, (l_dis-r_dis)/1.5))

    particles[:, 0] += update[0]*-np.sin(particles[:,2]) + (randn(N)*std[1])
    particles[:, 1] += update[1]*np.cos(particles[:,2]) + (randn(N)*std[1])
    particles[:, 2] += update[2] + (randn(N) * std[0])
    particles[:, 2] %= 2 * np.pi


def update(particles, weights, z, R, landmarks):
    for i, landmark in enumerate(landmarks):
        distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
        weights *= scipy.stats.norm(distance, R).pdf(z[i])

    weights += 1.e-300      # avoid round-off to zero
    weights /= sum(weights) # normalize


def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""
    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var  = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var

def neff(weights):
    return 1. / np.sum(np.square(weights))

def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights.fill(1.0 / len(weights))

