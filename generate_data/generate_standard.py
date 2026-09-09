import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def circle():
    # generate n points within the circle of radius r centered at x
    n = 400
    center = [0.5, 0.5]  # center
    radius = 0.5  # radius

    theta = np.random.choice(np.array([np.pi/2, 3*np.pi/2]), n)#np.pi * np.random.rand(n) + np.pi/2
    rho = np.sqrt(np.random.rand(n))

    polar = np.vstack((theta, rho)).T

    x = np.vstack((rho * np.cos(theta), rho * np.sin(theta))).T

    x = x * radius + np.array(center)

    # x = np.concatenate((theta.reshape(-1,1), rho.reshape(-1,1)), axis = 1)
    np.save('../data/polar_line/circle_xy_interface_400.npy', x)
    #np.save('data/polar_line/data_random_unit_circle_interface.npy', polar)

    plt.scatter(x[:, 0], x[:, 1])
    axis = plt.gca()
    axis.set_aspect(1)
    plt.show()

def circle_boundary():
    n = 200
    center = [0.5, 0.5]  # center
    radius = 0.5  # radius

    theta = (2 * np.pi / n) * np.arange(n).reshape(-1)- np.pi/2
    rho = np.ones_like(theta)

    polar = np.vstack((theta, rho)).T

    x = np.vstack((rho * np.cos(theta), rho * np.sin(theta))).T

    x = x * radius + np.array(center)

    np.save('../data/polar_line/circle_boundary', x)

    plt.scatter(x[:, 0], x[:, 1])
    axis = plt.gca()
    axis.set_aspect(1)
    plt.show()

def uniform_mesh(n):
    x = np.linspace(0, 1, num=n)
    points = np.array(list(product(x, x)))
    #np.save('../data/square_poly/points_uniform', points)
    return points

def main():
    pts = uniform_mesh(101)
    np.save('../data/square/points_uniform', pts)

if __name__ == '__main__':
    main()