import numpy as np

def normal(vertex, f, c, i, n):
    a = np.zeros((4, 4 * (n - 1) ** 2))
    b = np.zeros((4))
    mu = np.sqrt(c[i])
    # left
    a[0][i], a[0][i - (n - 1)] = np.exp(mu * vertex[0][0]), - np.exp(mu * vertex[0][0])
    a[0][(n - 1) ** 2 + i], a[0][(n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][0]), - np.exp(
        -mu * vertex[0][0])
    a[0][2 * (n - 1) ** 2 + i], a[0][2 * (n - 1) ** 2 + i - (n - 1)] = np.exp(mu * vertex[0][1]), - np.exp(
        mu * vertex[0][1])
    a[0][3 * (n - 1) ** 2 + i], a[0][3 * (n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][1]), - np.exp(
        -mu * vertex[0][1])
    b[0] = f[i - (n - 1)] - f[i]  # later
    b[0] = f[i - (n - 1)] - f[i]  # later
    a[1][i], a[1][i - (n - 1)] = -np.exp(mu * vertex[0][0]), np.exp(mu * vertex[0][0])
    a[1][(n - 1) ** 2 + i], a[1][(n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][0]), -np.exp(
        -mu * vertex[0][0])
    # up
    a[2][i], a[2][i + 1] = np.exp(mu * vertex[1][0]), - np.exp(mu * vertex[1][0])
    a[2][(n - 1) ** 2 + i], a[2][(n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][0]), - np.exp(
        -mu * vertex[1][0])
    a[2][2 * (n - 1) ** 2 + i], a[2][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), - np.exp(
        mu * vertex[1][1])
    a[2][3 * (n - 1) ** 2 + i], a[2][3 * (n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][1]), - np.exp(
        -mu * vertex[1][1])
    b[2] = f[i + 1] - f[i]  # later
    a[3][2 * (n - 1) ** 2 + i], a[3][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), - np.exp(
        mu * vertex[1][1])
    a[3][3 * (n - 1) ** 2 + i], a[3][3 * (n - 1) ** 2 + i + 1] = -np.exp(-mu * vertex[1][1]), + np.exp(
        -mu * vertex[1][1])
    return a, b

def left_outer(vertex,f, c, i, n):
    a = np.zeros((3,4*(n-1)**2))
    b = np.zeros((3))
    mu = np.sqrt(c[i])
    # left
    a[0][i] = np.exp(mu * vertex[0][0])
    a[0][(n - 1) ** 2 + i] = np.exp(-mu * vertex[0][0])
    a[0][2 * (n - 1) ** 2 + i] = np.exp(mu * vertex[0][1])
    a[0][3 * (n - 1) ** 2 + i] = np.exp(-mu * vertex[0][1])
    b[0] = -f[i]
    # up
    a[1][i], a[1][i + 1] = np.exp(mu * vertex[1][0]), - np.exp(mu * vertex[1][0])
    a[1][(n - 1) ** 2 + i], a[1][(n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][0]), - np.exp(-mu * vertex[1][0])
    a[1][2 * (n - 1) ** 2 + i], a[1][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), - np.exp(mu * vertex[1][1])
    a[1][3 * (n - 1) ** 2 + i], a[1][3 * (n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][1]), -np.exp(-mu * vertex[1][1])
    b[1] = f[i + 1] - f[i]  # later
    a[2][2 * (n - 1) ** 2 + i], a[2][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), - np.exp(mu * vertex[1][1])
    a[2][3 * (n - 1) ** 2 + i], a[2][3 * (n - 1) ** 2 + i + 1] = -np.exp(-mu * vertex[1][1]), np.exp(-mu * vertex[1][1])
    return a, b

def up_outer(vertex,f,c, i, n):
    a = np.zeros((3,4*(n-1)**2))
    b = np.zeros((3))
    mu = np.sqrt(c[i])
    # up
    a[0][i] = np.exp(mu*vertex[1][0])
    a[0][(n-1)**2 + i] = np.exp(-mu*vertex[1][0])
    a[0][2 * (n-1)**2 + i] = np.exp(mu*vertex[1][1])
    a[0][3 * (n-1)**2 + i] = np.exp(-mu*vertex[1][1])
    b[0] = -f[i]
    # left
    a[1][i], a[1][i - (n-1)] = np.exp(mu*vertex[0][0]), - np.exp(mu*vertex[0][0])
    a[1][(n-1)**2 + i], a[1][(n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][0]), - np.exp(-mu*vertex[0][0])
    a[1][2 * (n-1)**2 + i], a[1][2 * (n-1)**2 + i - (n-1)] = np.exp(mu*vertex[0][1]), - np.exp(mu*vertex[0][1])
    a[1][3 * (n-1)**2 + i], a[1][3 * (n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][1]), - np.exp(-mu*vertex[0][1])
    b[1] = f[i - (n-1)] - f[i]  # later
    a[2][i], a[2][i - (n-1)] = -np.exp(mu*vertex[0][0]),  np.exp(mu*vertex[0][0])
    a[2][(n-1)**2 + i], a[2][(n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][0]),  -np.exp(-mu*vertex[0][0])
    return a, b

def right_outer(vertex,f, c, i, n):
    a = np.zeros((5,4*(n-1)**2))
    b = np.zeros((5))
    mu = np.sqrt(c[i])
    # right
    a[0][i] = np.exp(mu * vertex[2][0])
    a[0][(n - 1) ** 2 + i] = np.exp(-mu * vertex[2][0])
    a[0][2 * (n - 1) ** 2 + i] = np.exp(mu * vertex[2][1])
    a[0][3 * (n - 1) ** 2 + i] = np.exp(-mu * vertex[2][1])
    b[0] = -f[i]
    # up
    a[1][i], a[1][i + 1] = np.exp(mu * vertex[1][0]), - np.exp(mu * vertex[1][0])
    a[1][(n - 1) ** 2 + i], a[1][(n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][0]), - np.exp(-mu * vertex[1][0])
    a[1][2 * (n - 1) ** 2 + i], a[1][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), - np.exp(mu * vertex[1][1])
    a[1][3 * (n - 1) ** 2 + i], a[1][3 * (n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][1]), - np.exp(
        -mu * vertex[1][1])
    b[1] = f[i + 1] - f[i]  # later
    a[2][2 * (n - 1) ** 2 + i], a[2][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), - np.exp(mu * vertex[1][1])
    a[2][3 * (n - 1) ** 2 + i], a[2][3 * (n - 1) ** 2 + i + 1] = -np.exp(-mu * vertex[1][1]), np.exp(-mu * vertex[1][1])
    # left
    a[3][i], a[3][i - (n - 1)] = np.exp(mu * vertex[0][0]), - np.exp(mu * vertex[0][0])
    a[3][(n - 1) ** 2 + i], a[3][(n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][0]), - np.exp(-mu * vertex[0][0])
    a[3][2 * (n - 1) ** 2 + i], a[3][2 * (n - 1) ** 2 + i - (n - 1)] = np.exp(mu * vertex[0][1]), - np.exp(
        mu * vertex[0][1])
    a[3][3 * (n - 1) ** 2 + i], a[3][3 * (n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][1]), - np.exp(
        -mu * vertex[0][1])
    b[3] = f[i - (n - 1)] - f[i]  # later
    a[4][i], a[4][i - (n - 1)] = -np.exp(mu * vertex[0][0]), np.exp(mu * vertex[0][0])
    a[4][(n - 1) ** 2 + i], a[4][(n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][0]), -np.exp(-mu * vertex[0][0])
    return a, b

def down_outer(vertex,f, c, i, n):
    a = np.zeros((5,4*(n-1)**2))
    b = np.zeros((5))
    mu = np.sqrt(c[i])
    # down
    a[0][i] = np.exp(mu * vertex[3][0])
    a[0][(n - 1) ** 2 + i] = np.exp(-mu * vertex[3][0])
    a[0][2 * (n - 1) ** 2 + i] = np.exp(mu * vertex[3][1])
    a[0][3 * (n - 1) ** 2 + i] = np.exp(-mu * vertex[3][1])
    b[0] = -f[i]
    # up
    a[1][i], a[1][i + 1] = np.exp(mu * vertex[1][0]), - np.exp(mu * vertex[1][0])
    a[1][(n - 1) ** 2 + i], a[1][(n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][0]), - np.exp(-mu * vertex[1][0])
    a[1][2 * (n - 1) ** 2 + i], a[1][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), - np.exp(mu * vertex[1][1])
    a[1][3 * (n - 1) ** 2 + i], a[1][3 * (n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][1]), - np.exp(
        -mu * vertex[1][1])
    b[1] = f[i + 1] - f[i]  # later
    a[2][2 * (n - 1) ** 2 + i], a[2][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), - np.exp(mu * vertex[1][1])
    a[2][3 * (n - 1) ** 2 + i], a[2][3 * (n - 1) ** 2 + i + 1] = -np.exp(-mu * vertex[1][1]), np.exp(-mu * vertex[1][1])
    # left
    a[3][i], a[3][i - (n - 1)] = np.exp(mu * vertex[0][0]), - np.exp(mu * vertex[0][0])
    a[3][(n - 1) ** 2 + i], a[3][(n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][0]), - np.exp(-mu * vertex[0][0])
    a[3][2 * (n - 1) ** 2 + i], a[3][2 * (n - 1) ** 2 + i - (n - 1)] = np.exp(mu * vertex[0][1]), - np.exp(
        mu * vertex[0][1])
    a[3][3 * (n - 1) ** 2 + i], a[3][3 * (n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][1]), - np.exp(
        -mu * vertex[0][1])
    b[3] = f[i - (n - 1)] - f[i]  # later
    a[4][i], a[4][i - (n - 1)] = -np.exp(mu * vertex[0][0]), np.exp(mu * vertex[0][0])
    a[4][(n - 1) ** 2 + i], a[4][(n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][0]), -np.exp(-mu * vertex[0][0])
    return a, b

def top(vertex, f, c, i, n):
    if i == 0:
        a = np.zeros((4, 4 * (n - 1) ** 2))
        b = np.zeros((4))
        mu = np.sqrt(c[i])
        # left
        a[0][i] = np.exp(mu * vertex[0][0])
        a[0][(n - 1) ** 2 + i] = np.exp(-mu * vertex[0][0])
        a[0][2 * (n - 1) ** 2 + i] = np.exp(mu * vertex[0][1])
        a[0][3 * (n - 1) ** 2 + i] = np.exp(-mu * vertex[0][1])
        b[0] = -f[i]
        # up
        a[1][i], a[1][i + 1] = np.exp(mu * vertex[1][0]), - np.exp(mu * vertex[1][0])
        a[1][(n - 1) ** 2 + i], a[1][(n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][0]), - np.exp(-mu * vertex[1][0])
        a[1][2 * (n - 1) ** 2 + i], a[1][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), - np.exp(
            mu * vertex[1][1])
        a[1][3 * (n - 1) ** 2 + i], a[1][3 * (n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][1]), - np.exp(
            -mu * vertex[1][1])
        b[1] = f[i + 1] - f[i]  # later
        a[2][2 * (n - 1) ** 2 + i], a[2][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), - np.exp(
            mu * vertex[1][1])
        a[2][3 * (n - 1) ** 2 + i], a[2][3 * (n - 1) ** 2 + i + 1] = -np.exp(-mu * vertex[1][1]), np.exp(
            -mu * vertex[1][1])
        # down
        a[3][i] = np.exp(mu * vertex[3][0])
        a[3][(n - 1) ** 2 + i] = np.exp(-mu * vertex[3][0])
        a[3][2 * (n - 1) ** 2 + i] = np.exp(mu * vertex[3][1])
        a[3][3 * (n - 1) ** 2 + i] = np.exp(-mu * vertex[3][1])
        b[3] = - f[i]  # later
        return a, b
    if i == n-2:
        a = np.zeros((2, 4 * (n - 1) ** 2))
        b = np.zeros((2))
        mu = np.sqrt(c[i])
        # left
        a[0][i] = np.exp(mu * vertex[0][0])
        a[0][(n - 1) ** 2 + i] = np.exp(-mu * vertex[0][0])
        a[0][2 * (n - 1) ** 2 + i] = np.exp(mu * vertex[0][1])
        a[0][3 * (n - 1) ** 2 + i] = np.exp(-mu * vertex[0][1])
        b[0] = -f[i]
        # up
        a[1][i] = np.exp(mu * vertex[1][0])
        a[1][(n - 1) ** 2 + i] = np.exp(-mu * vertex[1][0])
        a[1][2 * (n - 1) ** 2 + i] = np.exp(mu * vertex[1][1])
        a[1][3 * (n - 1) ** 2 + i] = np.exp(-mu * vertex[1][1])
        b[1] = - f[i]  # later
        return a, b
    if i == (n - 1) * (n - 2):
        a = np.zeros((6, 4 * (n - 1) ** 2))
        b = np.zeros((6))
        mu = np.sqrt(c[i])
        # right
        a[0][i] = np.exp(mu * vertex[2][0])
        a[0][(n - 1) ** 2 + i] = np.exp(-mu * vertex[2][0])
        a[0][2 * (n - 1) ** 2 + i] = np.exp(mu * vertex[2][1])
        a[0][3 * (n - 1) ** 2 + i] = np.exp(-mu * vertex[2][1])
        b[0] = -f[i]
        # up
        a[1][i], a[1][i + 1] = np.exp(mu * vertex[1][0]), - np.exp(mu * vertex[1][0])
        a[1][(n - 1) ** 2 + i], a[1][(n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][0]), - np.exp(-mu * vertex[1][0])
        a[1][2 * (n - 1) ** 2 + i], a[1][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), - np.exp(
            mu * vertex[1][1])
        a[1][3 * (n - 1) ** 2 + i], a[1][3 * (n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][1]), - np.exp(
            -mu * vertex[1][1])
        b[1] = f[i + 1] - f[i]  # later
        a[2][2 * (n - 1) ** 2 + i], a[2][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), -np.exp(
            mu * vertex[1][1])
        a[2][3 * (n - 1) ** 2 + i], a[2][3 * (n - 1) ** 2 + i + 1] = -np.exp(-mu * vertex[1][1]), np.exp(
            -mu * vertex[1][1])
        # left
        a[3][i], a[3][i - (n - 1)] = np.exp(mu * vertex[0][0]), - np.exp(mu * vertex[0][0])
        a[3][(n - 1) ** 2 + i], a[3][(n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][0]), - np.exp(
            -mu * vertex[0][0])
        a[3][2 * (n - 1) ** 2 + i], a[3][2 * (n - 1) ** 2 + i - (n - 1)] = np.exp(mu * vertex[0][1]), - np.exp(
            mu * vertex[0][1])
        a[3][3 * (n - 1) ** 2 + i], a[3][3 * (n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][1]), - np.exp(
            -mu * vertex[0][1])
        b[3] = f[i - (n - 1)] - f[i]  # later
        a[4][i], a[4][i - (n - 1)] = -np.exp(mu * vertex[0][0]), np.exp(mu * vertex[0][0])
        a[4][(n - 1) ** 2 + i], a[4][(n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][0]), -np.exp(
            -mu * vertex[0][0])
        # down
        a[5][i] = np.exp(mu * vertex[3][0])
        a[5][(n - 1) ** 2 + i] = np.exp(-mu * vertex[3][0])
        a[5][2 * (n - 1) ** 2 + i] = np.exp(mu * vertex[3][1])
        a[5][3 * (n - 1) ** 2 + i] = np.exp(-mu * vertex[3][1])
        b[5] = - f[i]  # later
        return a, b
    if i == n * (n - 2):
        a = np.zeros((4, 4 * (n - 1) ** 2))
        b = np.zeros((4))
        mu = np.sqrt(c[i])
        # right
        a[0][i] = np.exp(mu * vertex[2][0])
        a[0][(n - 1) ** 2 + i] = np.exp(-mu * vertex[2][0])
        a[0][2 * (n - 1) ** 2 + i] = np.exp(mu * vertex[2][1])
        a[0][3 * (n - 1) ** 2 + i] = np.exp(-mu * vertex[2][1])
        b[0] = -f[i]
        # up
        a[1][i] = np.exp(mu * vertex[1][0])
        a[1][(n - 1) ** 2 + i] = np.exp(-mu * vertex[1][0])
        a[1][2 * (n - 1) ** 2 + i] = np.exp(mu * vertex[1][1])
        a[1][3 * (n - 1) ** 2 + i] = np.exp(-mu * vertex[1][1])
        b[1] = - f[i]  # later
        # left
        a[2][i], a[2][i - (n - 1)] = np.exp(mu * vertex[0][0]), - np.exp(mu * vertex[0][0])
        a[2][(n - 1) ** 2 + i], a[2][(n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][0]), - np.exp(
            -mu * vertex[0][0])
        a[2][2 * (n - 1) ** 2 + i], a[2][2 * (n - 1) ** 2 + i - (n - 1)] = np.exp(mu * vertex[0][1]), - np.exp(
            mu * vertex[0][1])
        a[2][3 * (n - 1) ** 2 + i], a[2][3 * (n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][1]), - np.exp(
            -mu * vertex[0][1])
        b[2] = f[i - (n - 1)] - f[i]  # later
        a[3][i], a[3][i - (n - 1)] = -np.exp(mu * vertex[0][0]), np.exp(mu * vertex[0][0])
        a[3][(n - 1) ** 2 + i], a[3][(n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][0]), -np.exp(
            -mu * vertex[0][0])
        return a, b

def inter_out_left(vertex, f, c, i, n, jd, jv):
    a = np.zeros((4,4*(n-1)**2))
    b = np.zeros((4))
    mu = np.sqrt(c[i])
    # left
    a[0][i], a[0][i - (n-1)] = np.exp(mu*vertex[0][0]), - np.exp(mu*vertex[0][0])
    a[0][(n-1)**2 + i], a[0][(n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][0]), - np.exp(-mu*vertex[0][0])
    a[0][2 * (n-1)**2 + i], a[0][2 * (n-1)**2 + i - (n-1)] = np.exp(mu*vertex[0][1]), - np.exp(mu*vertex[0][1])
    a[0][3 * (n-1)**2 + i], a[0][3 * (n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][1]), - np.exp(-mu*vertex[0][1])
    b[0] = f[i - (n-1)] - f[i]  # later
    a[1][i], a[1][i - (n-1)] = -np.exp(mu*vertex[0][0]),  np.exp(mu*vertex[0][0])
    a[1][(n-1)**2 + i], a[1][(n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][0]), -np.exp(-mu*vertex[0][0])
    # up
    a[2][i], a[2][i + 1] = np.exp(mu*vertex[1][0]), - np.exp(mu*vertex[1][0])
    a[2][(n-1)**2 + i], a[2][(n-1)**2 + i + 1] = np.exp(-mu*vertex[1][0]), - np.exp(-mu*vertex[1][0])
    a[2][2 * (n-1)**2 + i], a[2][2 * (n-1)**2 + i + 1] = np.exp(mu*vertex[1][1]), - np.exp(mu*vertex[1][1])
    a[2][3 * (n-1)**2 + i], a[2][3 * (n-1)**2 + i + 1] = np.exp(-mu*vertex[1][1]), - np.exp(-mu*vertex[1][1])
    b[2] = f[i + 1] - f[i]  # later
    a[3][2 * (n-1)**2 + i], a[3][2 * (n-1)**2 + i + 1] = np.exp(mu*vertex[1][1]), - np.exp(mu*vertex[1][1])
    a[3][3 * (n-1)**2 + i], a[3][3 * (n-1)**2 + i + 1] = -np.exp(-mu*vertex[1][1]), + np.exp(-mu*vertex[1][1])
    return a, b

def inter_out_up(vertex, f, c, i, n, jd, jv):
    a = np.zeros((4,4*(n-1)**2))
    b = np.zeros((4))
    mu = np.sqrt(c[i])
    # left
    a[0][i], a[0][i - (n-1)] = np.exp(mu*vertex[0][0]), - np.exp(mu*vertex[0][0])
    a[0][(n-1)**2 + i], a[0][(n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][0]), - np.exp(-mu*vertex[0][0])
    a[0][2 * (n-1)**2 + i], a[0][2 * (n-1)**2 + i - (n-1)] = np.exp(mu*vertex[0][1]), - np.exp(mu*vertex[0][1])
    a[0][3 * (n-1)**2 + i], a[0][3 * (n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][1]), - np.exp(-mu*vertex[0][1])
    b[0] = f[i - (n-1)] - f[i]  # later
    a[1][i], a[1][i - (n-1)] = -np.exp(mu*vertex[0][0]),  np.exp(mu*vertex[0][0])
    a[1][(n-1)**2 + i], a[1][(n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][0]), -np.exp(-mu*vertex[0][0])
    # up
    a[2][i], a[2][i + 1] = np.exp(mu*vertex[1][0]), - np.exp(mu*vertex[1][0])
    a[2][(n-1)**2 + i], a[2][(n-1)**2 + i + 1] = np.exp(-mu*vertex[1][0]), - np.exp(-mu*vertex[1][0])
    a[2][2 * (n-1)**2 + i], a[2][2 * (n-1)**2 + i + 1] = np.exp(mu*vertex[1][1]), - np.exp(mu*vertex[1][1])
    a[2][3 * (n-1)**2 + i], a[2][3 * (n-1)**2 + i + 1] = np.exp(-mu*vertex[1][1]), - np.exp(-mu*vertex[1][1])
    b[2] = f[i + 1] - f[i]  # later
    a[3][2 * (n-1)**2 + i], a[3][2 * (n-1)**2 + i + 1] = np.exp(mu*vertex[1][1]), - np.exp(mu*vertex[1][1])
    a[3][3 * (n-1)**2 + i], a[3][3 * (n-1)**2 + i + 1] = -np.exp(-mu*vertex[1][1]), + np.exp(-mu*vertex[1][1])
    return a, b

def inter_out_right(vertex, f, c, i, n, jd, jv):
    a = np.zeros((4,4*(n-1)**2))
    b = np.zeros((4))
    mu = np.sqrt(c[i])
    # left
    a[0][i], a[0][i - (n-1)] = np.exp(mu*vertex[0][0]), - np.exp(mu*vertex[0][0])
    a[0][(n-1)**2 + i], a[0][(n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][0]), - np.exp(-mu*vertex[0][0])
    a[0][2 * (n-1)**2 + i], a[0][2 * (n-1)**2 + i - (n-1)] = np.exp(mu*vertex[0][1]), - np.exp(mu*vertex[0][1])
    a[0][3 * (n-1)**2 + i], a[0][3 * (n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][1]), - np.exp(-mu*vertex[0][1])
    b[0] = f[i - (n-1)] - f[i] + jv[i]  # later
    a[1][i], a[1][i - (n-1)] = np.exp(mu*vertex[0][0]), - np.exp(mu*vertex[0][0])
    a[1][(n-1)**2 + i], a[1][(n-1)**2 + i - (n-1)] = -np.exp(-mu*vertex[0][0]), np.exp(-mu*vertex[0][0])
    b[1] = jd[i]
    # up
    a[2][i], a[2][i + 1] = np.exp(mu*vertex[1][0]), - np.exp(mu*vertex[1][0])
    a[2][(n-1)**2 + i], a[2][(n-1)**2 + i + 1] = np.exp(-mu*vertex[1][0]), - np.exp(-mu*vertex[1][0])
    a[2][2 * (n-1)**2 + i], a[2][2 * (n-1)**2 + i + 1] = np.exp(mu*vertex[1][1]), - np.exp(mu*vertex[1][1])
    a[2][3 * (n-1)**2 + i], a[2][3 * (n-1)**2 + i + 1] = np.exp(-mu*vertex[1][1]), - np.exp(-mu*vertex[1][1])
    b[2] = f[i + 1] - f[i]  # later
    a[3][2 * (n-1)**2 + i], a[3][2 * (n-1)**2 + i + 1] = np.exp(mu*vertex[1][1]), - np.exp(mu*vertex[1][1])
    a[3][3 * (n-1)**2 + i], a[3][3 * (n-1)**2 + i + 1] = -np.exp(-mu*vertex[1][1]), + np.exp(-mu*vertex[1][1])
    return a, b

def inter_out_down(vertex, f, c, i, n, jd, jv):
    a = np.zeros((4,4*(n-1)**2))
    b = np.zeros((4))
    mu = np.sqrt(c[i])
    # left
    a[0][i], a[0][i - (n-1)] = np.exp(mu*vertex[0][0]), - np.exp(mu*vertex[0][0])
    a[0][(n-1)**2 + i], a[0][(n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][0]), - np.exp(-mu*vertex[0][0])
    a[0][2 * (n-1)**2 + i], a[0][2 * (n-1)**2 + i - (n-1)] = np.exp(mu*vertex[0][1]), - np.exp(mu*vertex[0][1])
    a[0][3 * (n-1)**2 + i], a[0][3 * (n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][1]), - np.exp(-mu*vertex[0][1])
    b[0] = f[i - (n-1)] - f[i]  # later
    a[1][i], a[1][i - (n-1)] = -np.exp(mu*vertex[0][0]), np.exp(mu*vertex[0][0])
    a[1][(n-1)**2 + i], a[1][(n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][0]), -np.exp(-mu*vertex[0][0])
    # up
    a[2][i], a[2][i + 1] = np.exp(mu*vertex[1][0]), - np.exp(mu*vertex[1][0])
    a[2][(n-1)**2 + i], a[2][(n-1)**2 + i + 1] = np.exp(-mu*vertex[1][0]), - np.exp(-mu*vertex[1][0])
    a[2][2 * (n-1)**2 + i], a[2][2 * (n-1)**2 + i + 1] = np.exp(mu*vertex[1][1]), - np.exp(mu*vertex[1][1])
    a[2][3 * (n-1)**2 + i], a[2][3 * (n-1)**2 + i + 1] = np.exp(-mu*vertex[1][1]), - np.exp(-mu*vertex[1][1])
    b[2] = f[i + 1] - f[i] + jv[i]  # later
    a[3][2 * (n-1)**2 + i], a[3][2 * (n-1)**2 + i + 1] = np.exp(mu*vertex[1][1]), - np.exp(mu*vertex[1][1])
    a[3][3 * (n-1)**2 + i], a[3][3 * (n-1)**2 + i + 1] = -np.exp(-mu*vertex[1][1]), + np.exp(-mu*vertex[1][1])
    b[3] = -jd[i]
    return a, b

def inter_in_left(vertex, f, c, i, n, jd, jv):
    a = np.zeros((4,4*(n-1)**2))
    b = np.zeros((4))
    mu = np.sqrt(c[i])
    # left
    a[0][i], a[0][i - (n-1)] = np.exp(mu*vertex[0][0]), - np.exp(mu*vertex[0][0])
    a[0][(n-1)**2 + i], a[0][(n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][0]), - np.exp(-mu*vertex[0][0])
    a[0][2 * (n-1)**2 + i], a[0][2 * (n-1)**2 + i - (n-1)] = np.exp(mu*vertex[0][1]), - np.exp(mu*vertex[0][1])
    a[0][3 * (n-1)**2 + i], a[0][3 * (n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][1]), - np.exp(-mu*vertex[0][1])
    b[0] = f[i - (n-1)] - f[i] - jv[i]  # later
    a[1][i], a[1][i - (n-1)] = np.exp(mu*vertex[0][0]), - np.exp(mu*vertex[0][0])
    a[1][(n-1)**2 + i], a[1][(n-1)**2 + i - (n-1)] = -np.exp(-mu*vertex[0][0]), np.exp(-mu*vertex[0][0])
    b[1] = jd[i]
    # up
    a[2][i], a[2][i + 1] = np.exp(mu*vertex[1][0]), - np.exp(mu*vertex[1][0])
    a[2][(n-1)**2 + i], a[2][(n-1)**2 + i + 1] = np.exp(-mu*vertex[1][0]), - np.exp(-mu*vertex[1][0])
    a[2][2 * (n-1)**2 + i], a[2][2 * (n-1)**2 + i + 1] = np.exp(mu*vertex[1][1]), - np.exp(mu*vertex[1][1])
    a[2][3 * (n-1)**2 + i], a[2][3 * (n-1)**2 + i + 1] = np.exp(-mu*vertex[1][1]), - np.exp(-mu*vertex[1][1])
    b[2] = f[i + 1] - f[i]  # later
    a[3][2 * (n-1)**2 + i], a[3][2 * (n-1)**2 + i + 1] = np.exp(mu*vertex[1][1]), - np.exp(mu*vertex[1][1])
    a[3][3 * (n-1)**2 + i], a[3][3 * (n-1)**2 + i + 1] = -np.exp(-mu*vertex[1][1]), + np.exp(-mu*vertex[1][1])
    return a, b

def inter_in_up(vertex, f, c, i, n, jd, jv):
    a = np.zeros((4,4*(n-1)**2))
    b = np.zeros((4))
    mu = np.sqrt(c[i])
    # left
    a[0][i], a[0][i - (n-1)] = np.exp(mu*vertex[0][0]), - np.exp(mu*vertex[0][0])
    a[0][(n-1)**2 + i], a[0][(n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][0]), - np.exp(-mu*vertex[0][0])
    a[0][2 * (n-1)**2 + i], a[0][2 * (n-1)**2 + i - (n-1)] = np.exp(mu*vertex[0][1]), - np.exp(mu*vertex[0][1])
    a[0][3 * (n-1)**2 + i], a[0][3 * (n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][1]), - np.exp(-mu*vertex[0][1])
    b[0] = f[i - (n-1)] - f[i]  # later
    a[1][i], a[1][i - (n-1)] = -np.exp(mu*vertex[0][0]), np.exp(mu*vertex[0][0])
    a[1][(n-1)**2 + i], a[1][(n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][0]), -np.exp(-mu*vertex[0][0])
    # up
    a[2][i], a[2][i + 1] = np.exp(mu*vertex[1][0]), - np.exp(mu*vertex[1][0])
    a[2][(n-1)**2 + i], a[2][(n-1)**2 + i + 1] = np.exp(-mu*vertex[1][0]), - np.exp(-mu*vertex[1][0])
    a[2][2 * (n-1)**2 + i], a[2][2 * (n-1)**2 + i + 1] = np.exp(mu*vertex[1][1]), - np.exp(mu*vertex[1][1])
    a[2][3 * (n-1)**2 + i], a[2][3 * (n-1)**2 + i + 1] = np.exp(-mu*vertex[1][1]), - np.exp(-mu*vertex[1][1])
    b[2] = f[i + 1] - f[i] - jv[i] # later
    a[3][2 * (n-1)**2 + i], a[3][2 * (n-1)**2 + i + 1] = np.exp(mu*vertex[1][1]), - np.exp(mu*vertex[1][1])
    a[3][3 * (n-1)**2 + i], a[3][3 * (n-1)**2 + i + 1] = -np.exp(-mu*vertex[1][1]), + np.exp(-mu*vertex[1][1])
    b[3] = -jd[i]
    return a, b

def inter_in_right(vertex, f, c, i, n, jd, jv):
    a = np.zeros((4,4*(n-1)**2))
    b = np.zeros((4))
    mu = np.sqrt(c[i])
    # left
    a[0][i], a[0][i - (n-1)] = np.exp(mu*vertex[0][0]), - np.exp(mu*vertex[0][0])
    a[0][(n-1)**2 + i], a[0][(n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][0]), - np.exp(-mu*vertex[0][0])
    a[0][2 * (n-1)**2 + i], a[0][2 * (n-1)**2 + i - (n-1)] = np.exp(mu*vertex[0][1]), - np.exp(mu*vertex[0][1])
    a[0][3 * (n-1)**2 + i], a[0][3 * (n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][1]), - np.exp(-mu*vertex[0][1])
    b[0] = f[i - (n-1)] - f[i]  # later
    a[1][i], a[1][i - (n-1)] = -np.exp(mu*vertex[0][0]), np.exp(mu*vertex[0][0])
    a[1][(n-1)**2 + i], a[1][(n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][0]), -np.exp(-mu*vertex[0][0])
    # up
    a[2][i], a[2][i + 1] = np.exp(mu*vertex[1][0]), - np.exp(mu*vertex[1][0])
    a[2][(n-1)**2 + i], a[2][(n-1)**2 + i + 1] = np.exp(-mu*vertex[1][0]), - np.exp(-mu*vertex[1][0])
    a[2][2 * (n-1)**2 + i], a[2][2 * (n-1)**2 + i + 1] = np.exp(mu*vertex[1][1]), - np.exp(mu*vertex[1][1])
    a[2][3 * (n-1)**2 + i], a[2][3 * (n-1)**2 + i + 1] = np.exp(-mu*vertex[1][1]), - np.exp(-mu*vertex[1][1])
    b[2] = f[i + 1] - f[i]  # later
    a[3][2 * (n-1)**2 + i], a[3][2 * (n-1)**2 + i + 1] = np.exp(mu*vertex[1][1]), - np.exp(mu*vertex[1][1])
    a[3][3 * (n-1)**2 + i], a[3][3 * (n-1)**2 + i + 1] = -np.exp(-mu*vertex[1][1]), + np.exp(-mu*vertex[1][1])
    return a, b

def inter_in_down(vertex, f, c, i, n, jd, jv):
    a = np.zeros((4,4*(n-1)**2))
    b = np.zeros((4))
    mu = np.sqrt(c[i])
    # left
    a[0][i], a[0][i - (n-1)] = np.exp(mu*vertex[0][0]), - np.exp(mu*vertex[0][0])
    a[0][(n-1)**2 + i], a[0][(n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][0]), - np.exp(-mu*vertex[0][0])
    a[0][2 * (n-1)**2 + i], a[0][2 * (n-1)**2 + i - (n-1)] = np.exp(mu*vertex[0][1]), - np.exp(mu*vertex[0][1])
    a[0][3 * (n-1)**2 + i], a[0][3 * (n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][1]), - np.exp(-mu*vertex[0][1])
    b[0] = f[i - (n-1)] - f[i]  # later
    a[1][i], a[1][i - (n-1)] = -np.exp(mu*vertex[0][0]), np.exp(mu*vertex[0][0])
    a[1][(n-1)**2 + i], a[1][(n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][0]), -np.exp(-mu*vertex[0][0])
    # up
    a[2][i], a[2][i + 1] = np.exp(mu*vertex[1][0]), - np.exp(mu*vertex[1][0])
    a[2][(n-1)**2 + i], a[2][(n-1)**2 + i + 1] = np.exp(-mu*vertex[1][0]), - np.exp(-mu*vertex[1][0])
    a[2][2 * (n-1)**2 + i], a[2][2 * (n-1)**2 + i + 1] = np.exp(mu*vertex[1][1]), - np.exp(mu*vertex[1][1])
    a[2][3 * (n-1)**2 + i], a[2][3 * (n-1)**2 + i + 1] = np.exp(-mu*vertex[1][1]), - np.exp(-mu*vertex[1][1])
    b[2] = f[i + 1] - f[i]  # later
    a[3][2 * (n-1)**2 + i], a[3][2 * (n-1)**2 + i + 1] = np.exp(mu*vertex[1][1]), - np.exp(mu*vertex[1][1])
    a[3][3 * (n-1)**2 + i], a[3][3 * (n-1)**2 + i + 1] = -np.exp(-mu*vertex[1][1]), + np.exp(-mu*vertex[1][1])
    return a, b

def inter_top(vertex, f, c, i, n, jd, jv, interface):
    h = 1 / (n - 1)
    ld = int((n-1) * interface[0][0]/h + interface[0][1]/h)
    mu = np.sqrt(c[i])
    if i == ld:
        a = np.zeros((4,4*(n-1)**2))
        b = np.zeros((4))
        # left
        a[0][i], a[0][i - (n-1)] = np.exp(mu*vertex[0][0]), - np.exp(mu*vertex[0][0])
        a[0][(n-1)**2 + i], a[0][(n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][0]), - np.exp(-mu*vertex[0][0])
        a[0][2 * (n-1)**2 + i], a[0][2 * (n-1)**2 + i - (n-1)] = np.exp(mu*vertex[0][1]), - np.exp(mu*vertex[0][1])
        a[0][3 * (n-1)**2 + i], a[0][3 * (n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][1]), - np.exp(-mu*vertex[0][1])
        b[0] = f[i - (n-1)] - f[i] - jv[i]  # later
        a[1][i], a[1][i - (n-1)] = np.exp(mu*vertex[0][0]), - np.exp(mu*vertex[0][0])
        a[1][(n-1)**2 + i], a[1][(n-1)**2 + i - (n-1)] = -np.exp(-mu*vertex[0][0]), np.exp(-mu*vertex[0][0])
        b[1] = jd[i]
        # up
        a[2][i], a[2][i + 1] = np.exp(mu*vertex[1][0]), - np.exp(mu*vertex[1][0])
        a[2][(n-1)**2 + i], a[2][(n-1)**2 + i + 1] = np.exp(-mu*vertex[1][0]), - np.exp(-mu*vertex[1][0])
        a[2][2 * (n-1)**2 + i], a[2][2 * (n-1)**2 + i + 1] = np.exp(mu*vertex[1][1]), - np.exp(mu*vertex[1][1])
        a[2][3 * (n-1)**2 + i], a[2][3 * (n-1)**2 + i + 1] = np.exp(-mu*vertex[1][1]), - np.exp(-mu*vertex[1][1])
        b[2] = f[i + 1] - f[i]  # later
        a[3][2 * (n-1)**2 + i], a[3][2 * (n-1)**2 + i + 1] = np.exp(mu*vertex[1][1]), - np.exp(mu*vertex[1][1])
        a[3][3 * (n-1)**2 + i], a[3][3 * (n-1)**2 + i + 1] = -np.exp(-mu*vertex[1][1]), + np.exp(-mu*vertex[1][1])
        return a, b
    if i == ld + int(0.4/h) -1:
        a = np.zeros((4,4*(n-1)**2))
        b = np.zeros((4))
        # left
        a[0][i], a[0][i - (n-1)] = np.exp(mu*vertex[0][0]), - np.exp(mu*vertex[0][0])
        a[0][(n-1)**2 + i], a[0][(n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][0]), - np.exp(-mu*vertex[0][0])
        a[0][2 * (n-1)**2 + i], a[0][2 * (n-1)**2 + i - (n-1)] = np.exp(mu*vertex[0][1]), - np.exp(mu*vertex[0][1])
        a[0][3 * (n-1)**2 + i], a[0][3 * (n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][1]), - np.exp(-mu*vertex[0][1])
        b[0] = f[i - (n-1)] - f[i] - jv[i]  # later
        a[1][i], a[1][i - (n-1)] = np.exp(mu*vertex[0][0]), - np.exp(mu*vertex[0][0])
        a[1][(n-1)**2 + i], a[1][(n-1)**2 + i - (n-1)] = -np.exp(-mu*vertex[0][0]), np.exp(-mu*vertex[0][0])
        b[1] = jd[i]
        # up
        a[2][i], a[2][i + 1] = np.exp(mu*vertex[1][0]), - np.exp(mu*vertex[1][0])
        a[2][(n-1)**2 + i], a[2][(n-1)**2 + i + 1] = np.exp(-mu*vertex[1][0]), - np.exp(-mu*vertex[1][0])
        a[2][2 * (n-1)**2 + i], a[2][2 * (n-1)**2 + i + 1] = np.exp(mu*vertex[1][1]), - np.exp(mu*vertex[1][1])
        a[2][3 * (n-1)**2 + i], a[2][3 * (n-1)**2 + i + 1] = np.exp(-mu*vertex[1][1]), - np.exp(-mu*vertex[1][1])
        b[2] = f[i + 1] - f[i] - jv[i]  # later
        a[3][2 * (n-1)**2 + i], a[3][2 * (n-1)**2 + i + 1] = np.exp(mu*vertex[1][1]), - np.exp(mu*vertex[1][1])
        a[3][3 * (n-1)**2 + i], a[3][3 * (n-1)**2 + i + 1] = -np.exp(-mu*vertex[1][1]), + np.exp(-mu*vertex[1][1])
        b[3] = -jd[i]
        return a, b
    if i == ld + (n-1) * (int(0.4/h) -1):
        a = np.zeros((4,4*(n-1)**2))
        b = np.zeros((4))
        # left
        a[0][i], a[0][i - (n-1)] = np.exp(mu*vertex[0][0]), - np.exp(mu*vertex[0][0])
        a[0][(n-1)**2 + i], a[0][(n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][0]), - np.exp(-mu*vertex[0][0])
        a[0][2 * (n-1)**2 + i], a[0][2 * (n-1)**2 + i - (n-1)] = np.exp(mu*vertex[0][1]), - np.exp(mu*vertex[0][1])
        a[0][3 * (n-1)**2 + i], a[0][3 * (n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][1]), - np.exp(-mu*vertex[0][1])
        b[0] = f[i - (n-1)] - f[i]  # later
        a[1][i], a[1][i - (n-1)] = -np.exp(mu*vertex[0][0]), np.exp(mu*vertex[0][0])
        a[1][(n-1)**2 + i], a[1][(n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][0]), -np.exp(-mu*vertex[0][0])
        # up
        a[2][i], a[2][i + 1] = np.exp(mu*vertex[1][0]), - np.exp(mu*vertex[1][0])
        a[2][(n-1)**2 + i], a[2][(n-1)**2 + i + 1] = np.exp(-mu*vertex[1][0]), - np.exp(-mu*vertex[1][0])
        a[2][2 * (n-1)**2 + i], a[2][2 * (n-1)**2 + i + 1] = np.exp(mu*vertex[1][1]), - np.exp(mu*vertex[1][1])
        a[2][3 * (n-1)**2 + i], a[2][3 * (n-1)**2 + i + 1] = np.exp(-mu*vertex[1][1]), - np.exp(-mu*vertex[1][1])
        b[2] = f[i + 1] - f[i]  # later
        a[3][2 * (n-1)**2 + i], a[3][2 * (n-1)**2 + i + 1] = np.exp(mu*vertex[1][1]), - np.exp(mu*vertex[1][1])
        a[3][3 * (n-1)**2 + i], a[3][3 * (n-1)**2 + i + 1] = -np.exp(-mu*vertex[1][1]), + np.exp(-mu*vertex[1][1])
        return a, b
    if i == ld + n * (int(0.4/h) -1):
        a = np.zeros((4,4*(n-1)**2))
        b = np.zeros((4))
        # left
        a[0][i], a[0][i - (n-1)] = np.exp(mu*vertex[0][0]), - np.exp(mu*vertex[0][0])
        a[0][(n-1)**2 + i], a[0][(n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][0]), - np.exp(-mu*vertex[0][0])
        a[0][2 * (n-1)**2 + i], a[0][2 * (n-1)**2 + i - (n-1)] = np.exp(mu*vertex[0][1]), - np.exp(mu*vertex[0][1])
        a[0][3 * (n-1)**2 + i], a[0][3 * (n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][1]), - np.exp(-mu*vertex[0][1])
        b[0] = f[i - (n-1)] - f[i]  # later
        a[1][i], a[1][i - (n-1)] = -np.exp(mu*vertex[0][0]), np.exp(mu*vertex[0][0])
        a[1][(n-1)**2 + i], a[1][(n-1)**2 + i - (n-1)] = np.exp(-mu*vertex[0][0]), -np.exp(-mu*vertex[0][0])
        # up
        a[2][i], a[2][i + 1] = np.exp(mu*vertex[1][0]), - np.exp(mu*vertex[1][0])
        a[2][(n-1)**2 + i], a[2][(n-1)**2 + i + 1] = np.exp(-mu*vertex[1][0]), - np.exp(-mu*vertex[1][0])
        a[2][2 * (n-1)**2 + i], a[2][2 * (n-1)**2 + i + 1] = np.exp(mu*vertex[1][1]), - np.exp(mu*vertex[1][1])
        a[2][3 * (n-1)**2 + i], a[2][3 * (n-1)**2 + i + 1] = np.exp(-mu*vertex[1][1]), - np.exp(-mu*vertex[1][1])
        b[2] = f[i + 1] - f[i] - jv[i]  # later
        a[3][2 * (n-1)**2 + i], a[3][2 * (n-1)**2 + i + 1] = np.exp(mu*vertex[1][1]), - np.exp(mu*vertex[1][1])
        a[3][3 * (n-1)**2 + i], a[3][3 * (n-1)**2 + i + 1] = -np.exp(-mu*vertex[1][1]), + np.exp(-mu*vertex[1][1])
        b[3] = -jd[i]
        return a,b


