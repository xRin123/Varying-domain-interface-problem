import numpy as np

def normal(vertex, f, a, b, i, n):
    A = np.zeros((4, 4 * (n - 1) ** 2))
    B = np.zeros((4))
    mu = np.sqrt(b[i]/a[i])
    mu_l = np.sqrt(b[i - (n - 1)]/a[i - (n - 1)])
    mu_u = np.sqrt(b[i + 1]/a[i + 1])
    # left
    A[0][i], A[0][i - (n - 1)] = np.exp(mu * vertex[0][0]), - np.exp(mu_l * vertex[0][0])
    A[0][(n - 1) ** 2 + i], A[0][(n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][0]), - np.exp(
        -mu_l * vertex[0][0])
    A[0][2 * (n - 1) ** 2 + i], A[0][2 * (n - 1) ** 2 + i - (n - 1)] = np.exp(mu * vertex[0][1]), - np.exp(
        mu_l * vertex[0][1])
    A[0][3 * (n - 1) ** 2 + i], A[0][3 * (n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][1]), - np.exp(
        -mu_l * vertex[0][1])
    B[0] = f[i - (n - 1)] - f[i]  # later
    A[1][i], A[1][i - (n - 1)] = -np.exp(mu * vertex[0][0]), np.exp(mu_l * vertex[0][0])
    A[1][(n - 1) ** 2 + i], A[1][(n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][0]), - np.exp(
        -mu_l * vertex[0][0])
    # up
    A[2][i], A[2][i + 1] = np.exp(mu * vertex[1][0]), - np.exp(mu_u * vertex[1][0])
    A[2][(n - 1) ** 2 + i], A[2][(n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][0]), - np.exp(
        -mu_u * vertex[1][0])
    A[2][2 * (n - 1) ** 2 + i], A[2][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), - np.exp(
        mu_u * vertex[1][1])
    A[2][3 * (n - 1) ** 2 + i], A[2][3 * (n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][1]), - np.exp(
        -mu_u * vertex[1][1])
    B[2] = f[i + 1] - f[i]  # lAter
    A[3][2 * (n - 1) ** 2 + i], A[3][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), - np.exp(
        mu_u * vertex[1][1])
    A[3][3 * (n - 1) ** 2 + i], A[3][3 * (n - 1) ** 2 + i + 1] = -np.exp(-mu * vertex[1][1]), + np.exp(
        -mu_u * vertex[1][1])
    return A, B

def left_outer(vertex, f, a, b, i, n):
    A = np.zeros((3,4*(n-1)**2))
    B = np.zeros((3))
    mu = np.sqrt(b[i] / a[i])
    mu_u = np.sqrt(b[i + 1] / a[i + 1])
    # left
    A[0][i] = np.exp(mu * vertex[0][0])
    A[0][(n - 1) ** 2 + i] = np.exp(-mu * vertex[0][0])
    A[0][2 * (n - 1) ** 2 + i] = np.exp(mu * vertex[0][1])
    A[0][3 * (n - 1) ** 2 + i] = np.exp(-mu * vertex[0][1])
    B[0] = -f[i]
    # up
    A[1][i], A[1][i + 1] = np.exp(mu * vertex[1][0]), - np.exp(mu_u * vertex[1][0])
    A[1][(n - 1) ** 2 + i], A[1][(n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][0]), - np.exp(
        -mu_u * vertex[1][0])
    A[1][2 * (n - 1) ** 2 + i], A[1][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), - np.exp(
        mu_u * vertex[1][1])
    A[1][3 * (n - 1) ** 2 + i], A[1][3 * (n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][1]), - np.exp(
        -mu_u * vertex[1][1])
    B[1] = f[i + 1] - f[i]
    A[2][2 * (n - 1) ** 2 + i], A[2][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), - np.exp(
        mu_u * vertex[1][1])
    A[2][3 * (n - 1) ** 2 + i], A[2][3 * (n - 1) ** 2 + i + 1] = - np.exp(-mu * vertex[1][1]), + np.exp(
        -mu_u * vertex[1][1])
    return A,B

def up_outer(vertex, f, a, b, i, n):
    A = np.zeros((3, 4 * (n - 1) ** 2))
    B = np.zeros((3))
    mu = np.sqrt(b[i] / a[i])
    mu_l = np.sqrt(b[i - (n - 1)] / a[i - (n - 1)])
    # up
    A[0][i] = np.exp(mu * vertex[1][0])
    A[0][(n - 1) ** 2 + i] = np.exp(-mu * vertex[1][0])
    A[0][2 * (n - 1) ** 2 + i] = np.exp(mu * vertex[1][1])
    A[0][3 * (n - 1) ** 2 + i] = np.exp(-mu * vertex[1][1])
    B[0] = -f[i]
    # left
    A[1][i], A[1][i - (n - 1)] = np.exp(mu * vertex[0][0]), - np.exp(mu_l * vertex[0][0])
    A[1][(n - 1) ** 2 + i], A[1][(n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][0]), - np.exp(
        -mu_l * vertex[0][0])
    A[1][2 * (n - 1) ** 2 + i], A[1][2 * (n - 1) ** 2 + i - (n - 1)] = np.exp(mu * vertex[0][1]), - np.exp(
        mu_l * vertex[0][1])
    A[1][3 * (n - 1) ** 2 + i], A[1][3 * (n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][1]), - np.exp(
        -mu_l * vertex[0][1])
    B[1] = f[i - (n - 1)] - f[i]  # lAter
    A[2][i], A[2][i - (n - 1)] = -np.exp(mu * vertex[0][0]), np.exp(mu_l * vertex[0][0])
    A[2][(n - 1) ** 2 + i], A[2][(n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][0]), -np.exp(-mu_l * vertex[0][0])
    return A,B

def right_outer(vertex, f, a, b, i, n):
    A = np.zeros((5, 4 * (n - 1) ** 2))
    B = np.zeros((5))
    mu = np.sqrt(b[i] / a[i])
    mu_l = np.sqrt(b[i - (n - 1)] / a[i - (n - 1)])
    mu_u = np.sqrt(b[i + 1] / a[i + 1])
    # right
    A[0][i] = np.exp(mu * vertex[2][0])
    A[0][(n - 1) ** 2 + i] = np.exp(-mu * vertex[2][0])
    A[0][2 * (n - 1) ** 2 + i] = np.exp(mu * vertex[2][1])
    A[0][3 * (n - 1) ** 2 + i] = np.exp(-mu * vertex[2][1])
    B[0] = -f[i]
    # up
    A[1][i], A[1][i + 1] = np.exp(mu * vertex[1][0]), - np.exp(mu_u * vertex[1][0])
    A[1][(n - 1) ** 2 + i], A[1][(n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][0]), - np.exp(-mu_u * vertex[1][0])
    A[1][2 * (n - 1) ** 2 + i], A[1][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), - np.exp(mu_u * vertex[1][1])
    A[1][3 * (n - 1) ** 2 + i], A[1][3 * (n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][1]), - np.exp(
        -mu_u * vertex[1][1])
    B[1] = f[i + 1] - f[i]  # lAter
    A[2][2 * (n - 1) ** 2 + i], A[2][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), - np.exp(mu_u * vertex[1][1])
    A[2][3 * (n - 1) ** 2 + i], A[2][3 * (n - 1) ** 2 + i + 1] = -np.exp(-mu * vertex[1][1]), np.exp(-mu_u * vertex[1][1])
    # left
    A[3][i], A[3][i - (n - 1)] = np.exp(mu * vertex[0][0]), - np.exp(mu_l * vertex[0][0])
    A[3][(n - 1) ** 2 + i], A[3][(n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][0]), - np.exp(-mu_l * vertex[0][0])
    A[3][2 * (n - 1) ** 2 + i], A[3][2 * (n - 1) ** 2 + i - (n - 1)] = np.exp(mu * vertex[0][1]), - np.exp(
        mu_l * vertex[0][1])
    A[3][3 * (n - 1) ** 2 + i], A[3][3 * (n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][1]), - np.exp(
        -mu_l * vertex[0][1])
    B[3] = f[i - (n - 1)] - f[i]  # lAter
    A[4][i], A[4][i - (n - 1)] = -np.exp(mu * vertex[0][0]), np.exp(mu_l * vertex[0][0])
    A[4][(n - 1) ** 2 + i], A[4][(n - 1) ** 2 + i - (n - 1)] =  np.exp(-mu * vertex[0][0]), -np.exp(-mu_l * vertex[0][0])
    return A, B

def down_outer(vertex, f, a, b, i, n):
    A = np.zeros((5,4*(n-1)**2))
    B = np.zeros((5))
    mu = np.sqrt(b[i] / a[i])
    mu_l = np.sqrt(b[i - (n - 1)] / a[i - (n - 1)])
    mu_u = np.sqrt(b[i + 1] / a[i + 1])
    #down
    A[0][i] = np.exp(mu * vertex[3][0])
    A[0][(n - 1) ** 2 + i] = np.exp(-mu * vertex[3][0])
    A[0][2 * (n - 1) ** 2 + i] = np.exp(mu * vertex[3][1])
    A[0][3 * (n - 1) ** 2 + i] = np.exp(-mu * vertex[3][1])
    B[0] = -f[i]
    # up
    A[1][i], A[1][i + 1] = np.exp(mu * vertex[1][0]), - np.exp(mu_u * vertex[1][0])
    A[1][(n - 1) ** 2 + i], A[1][(n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][0]), - np.exp(-mu_u * vertex[1][0])
    A[1][2 * (n - 1) ** 2 + i], A[1][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), - np.exp(mu_u * vertex[1][1])
    A[1][3 * (n - 1) ** 2 + i], A[1][3 * (n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][1]), - np.exp(
        -mu_u * vertex[1][1])
    B[1] = f[i + 1] - f[i]  # lAter
    A[2][2 * (n - 1) ** 2 + i], A[2][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), - np.exp(mu_u * vertex[1][1])
    A[2][3 * (n - 1) ** 2 + i], A[2][3 * (n - 1) ** 2 + i + 1] = -np.exp(-mu * vertex[1][1]), np.exp(-mu_u * vertex[1][1])
    # left
    A[3][i], A[3][i - (n - 1)] = np.exp(mu * vertex[0][0]), - np.exp(mu_l * vertex[0][0])
    A[3][(n - 1) ** 2 + i], A[3][(n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][0]), - np.exp(-mu_l * vertex[0][0])
    A[3][2 * (n - 1) ** 2 + i], A[3][2 * (n - 1) ** 2 + i - (n - 1)] = np.exp(mu * vertex[0][1]), - np.exp(
        mu_l * vertex[0][1])
    A[3][3 * (n - 1) ** 2 + i], A[3][3 * (n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][1]), - np.exp(
        -mu_l * vertex[0][1])
    B[3] = f[i - (n - 1)] - f[i]  # lAter
    A[4][i], A[4][i - (n - 1)] = - np.exp(mu * vertex[0][0]), np.exp(mu_l * vertex[0][0])
    A[4][(n - 1) ** 2 + i], A[4][(n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][0]), -np.exp(-mu_l * vertex[0][0])
    return A, B

def top(vertex, f, a, b, i, n):
    if i == 0:
        A = np.zeros((4, 4 * (n - 1) ** 2))
        B = np.zeros((4))
        mu = np.sqrt(b[i] / a[i])
        mu_u = np.sqrt(b[i + 1] / a[i + 1])
        # left
        A[0][i] = np.exp(mu * vertex[0][0])
        A[0][(n - 1) ** 2 + i] = np.exp(-mu * vertex[0][0])
        A[0][2 * (n - 1) ** 2 + i] = np.exp(mu * vertex[0][1])
        A[0][3 * (n - 1) ** 2 + i] = np.exp(-mu * vertex[0][1])
        B[0] = -f[i]
        # up
        A[1][i], A[1][i + 1] = np.exp(mu * vertex[1][0]), - np.exp(mu_u * vertex[1][0])
        A[1][(n - 1) ** 2 + i], A[1][(n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][0]), - np.exp(-mu_u * vertex[1][0])
        A[1][2 * (n - 1) ** 2 + i], A[1][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), - np.exp(
            mu_u * vertex[1][1])
        A[1][3 * (n - 1) ** 2 + i], A[1][3 * (n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][1]), - np.exp(
            -mu_u * vertex[1][1])
        B[1] = f[i + 1] - f[i]  # lAter
        A[2][2 * (n - 1) ** 2 + i], A[2][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), - np.exp(
            mu_u * vertex[1][1])
        A[2][3 * (n - 1) ** 2 + i], A[2][3 * (n - 1) ** 2 + i + 1] = -np.exp(-mu * vertex[1][1]), np.exp(
            -mu_u * vertex[1][1])
        # down
        A[3][i] = np.exp(mu * vertex[3][0])
        A[3][(n - 1) ** 2 + i] = np.exp(-mu * vertex[3][0])
        A[3][2 * (n - 1) ** 2 + i] = np.exp(mu * vertex[3][1])
        A[3][3 * (n - 1) ** 2 + i] = np.exp(-mu * vertex[3][1])
        B[3] = - f[i]  # lAter
        return A, B
    if i == n - 2:
        A = np.zeros((2, 4 * (n - 1) ** 2))
        B = np.zeros((2))
        mu = np.sqrt(b[i] / a[i])
        # left
        A[0][i] = np.exp(mu * vertex[0][0])
        A[0][(n - 1) ** 2 + i] = np.exp(-mu * vertex[0][0])
        A[0][2 * (n - 1) ** 2 + i] = np.exp(mu * vertex[0][1])
        A[0][3 * (n - 1) ** 2 + i] = np.exp(-mu * vertex[0][1])
        B[0] = -f[i]
        # up
        A[1][i] = np.exp(mu * vertex[1][0])
        A[1][(n - 1) ** 2 + i] = np.exp(-mu * vertex[1][0])
        A[1][2 * (n - 1) ** 2 + i] = np.exp(mu * vertex[1][1])
        A[1][3 * (n - 1) ** 2 + i] = np.exp(-mu * vertex[1][1])
        B[1] = - f[i]  # lAter
        return A, B
    if i == (n - 1) * (n - 2):
        A = np.zeros((6, 4 * (n - 1) ** 2))
        B = np.zeros((6))
        mu = np.sqrt(b[i] / a[i])
        mu_l = np.sqrt(b[i - (n - 1)] / a[i - (n - 1)])
        mu_u = np.sqrt(b[i + 1] / a[i + 1])
        # right
        A[0][i] = np.exp(mu * vertex[2][0])
        A[0][(n - 1) ** 2 + i] = np.exp(-mu * vertex[2][0])
        A[0][2 * (n - 1) ** 2 + i] = np.exp(mu * vertex[2][1])
        A[0][3 * (n - 1) ** 2 + i] = np.exp(-mu * vertex[2][1])
        B[0] = -f[i]
        # up
        A[1][i], A[1][i + 1] = np.exp(mu * vertex[1][0]), - np.exp(mu_u * vertex[1][0])
        A[1][(n - 1) ** 2 + i], A[1][(n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][0]), - np.exp(
            -mu_u * vertex[1][0])
        A[1][2 * (n - 1) ** 2 + i], A[1][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), - np.exp(
            mu_u * vertex[1][1])
        A[1][3 * (n - 1) ** 2 + i], A[1][3 * (n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][1]), - np.exp(
            -mu_u * vertex[1][1])
        B[1] = f[i + 1] - f[i]  # lAter
        A[2][2 * (n - 1) ** 2 + i], A[2][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), - np.exp(
            mu_u * vertex[1][1])
        A[2][3 * (n - 1) ** 2 + i], A[2][3 * (n - 1) ** 2 + i + 1] = -np.exp(-mu * vertex[1][1]), + np.exp(
            -mu_u * vertex[1][1])
        # left
        A[3][i], A[3][i - (n - 1)] = np.exp(mu * vertex[0][0]), - np.exp(mu_l * vertex[0][0])
        A[3][(n - 1) ** 2 + i], A[3][(n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][0]), - np.exp(
            -mu_l * vertex[0][0])
        A[3][2 * (n - 1) ** 2 + i], A[3][2 * (n - 1) ** 2 + i - (n - 1)] = np.exp(mu * vertex[0][1]), - np.exp(
            mu_l * vertex[0][1])
        A[3][3 * (n - 1) ** 2 + i], A[3][3 * (n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][1]), - np.exp(
            -mu_l * vertex[0][1])
        B[3] = f[i - (n - 1)] - f[i]  # later
        A[4][i], A[4][i - (n - 1)] = - np.exp(mu * vertex[0][0]), np.exp(mu_l * vertex[0][0])
        A[4][(n - 1) ** 2 + i], A[4][(n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][0]), - np.exp(
            -mu_l * vertex[0][0])
        # down
        A[5][i] = np.exp(mu * vertex[3][0])
        A[5][(n - 1) ** 2 + i] = np.exp(-mu * vertex[3][0])
        A[5][2 * (n - 1) ** 2 + i] = np.exp(mu * vertex[3][1])
        A[5][3 * (n - 1) ** 2 + i] = np.exp(-mu * vertex[3][1])
        B[5] = - f[i]  # lAter
        return A, B
    if i == n * (n - 2):
        A = np.zeros((4, 4 * (n - 1) ** 2))
        B = np.zeros((4))
        mu = np.sqrt(b[i] / a[i])
        mu_l = np.sqrt(b[i - (n - 1)] / a[i - (n - 1)])
        # right
        A[0][i] = np.exp(mu * vertex[2][0])
        A[0][(n - 1) ** 2 + i] = np.exp(-mu * vertex[2][0])
        A[0][2 * (n - 1) ** 2 + i] = np.exp(mu * vertex[2][1])
        A[0][3 * (n - 1) ** 2 + i] = np.exp(-mu * vertex[2][1])
        B[0] = -f[i]
        # up
        A[1][i] = np.exp(mu * vertex[1][0])
        A[1][(n - 1) ** 2 + i] = np.exp(-mu * vertex[1][0])
        A[1][2 * (n - 1) ** 2 + i] = np.exp(mu * vertex[1][1])
        A[1][3 * (n - 1) ** 2 + i] = np.exp(-mu * vertex[1][1])
        B[1] = - f[i]  # lAter
        # left
        A[2][i], A[2][i - (n - 1)] = np.exp(mu * vertex[0][0]), - np.exp(mu_l * vertex[0][0])
        A[2][(n - 1) ** 2 + i], A[2][(n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][0]), - np.exp(
            -mu_l * vertex[0][0])
        A[2][2 * (n - 1) ** 2 + i], A[2][2 * (n - 1) ** 2 + i - (n - 1)] = np.exp(mu * vertex[0][1]), - np.exp(
            mu_l * vertex[0][1])
        A[2][3 * (n - 1) ** 2 + i], A[2][3 * (n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][1]), - np.exp(
            -mu_l * vertex[0][1])
        B[2] = f[i - (n - 1)] - f[i]  # later
        A[3][i], A[3][i - (n - 1)] = - np.exp(mu * vertex[0][0]), np.exp(mu_l * vertex[0][0])
        A[3][(n - 1) ** 2 + i], A[3][(n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][0]), - np.exp(
            -mu_l * vertex[0][0])
        return A, B

def inter_right(vertex, f, a, b, i, n, jd, jv):
    A = np.zeros((4, 4 * (n - 1) ** 2))
    B = np.zeros((4))
    mu = np.sqrt(b[i] / a[i])
    mu_l = np.sqrt(b[i - (n - 1)] / a[i - (n - 1)])
    mu_u = np.sqrt(b[i + 1] / a[i + 1])
    # left
    A[0][i], A[0][i - (n - 1)] = np.exp(mu * vertex[0][0]), - np.exp(mu_l * vertex[0][0])
    A[0][(n - 1) ** 2 + i], A[0][(n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][0]), - np.exp(-mu_l * vertex[0][0])
    A[0][2 * (n - 1) ** 2 + i], A[0][2 * (n - 1) ** 2 + i - (n - 1)] = np.exp(mu * vertex[0][1]), - np.exp(
        mu_l * vertex[0][1])
    A[0][3 * (n - 1) ** 2 + i], A[0][3 * (n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][1]), - np.exp(
        -mu_l * vertex[0][1])
    B[0] = f[i - (n - 1)] - f[i] + jv[i]  # lAter
    A[1][i], A[1][i - (n - 1)] = np.exp(mu * vertex[0][0]), - np.exp(mu_l * vertex[0][0])
    A[1][(n - 1) ** 2 + i], A[1][(n - 1) ** 2 + i - (n - 1)] = -np.exp(-mu * vertex[0][0]), np.exp(-mu_l * vertex[0][0])
    B[1] = jd[i]
    # up
    A[2][i], A[2][i + 1] = np.exp(mu * vertex[1][0]), - np.exp(mu_u * vertex[1][0])
    A[2][(n - 1) ** 2 + i], A[2][(n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][0]), - np.exp(-mu_u * vertex[1][0])
    A[2][2 * (n - 1) ** 2 + i], A[2][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), - np.exp(mu_u * vertex[1][1])
    A[2][3 * (n - 1) ** 2 + i], A[2][3 * (n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][1]), - np.exp(
        -mu_u * vertex[1][1])
    B[2] = f[i + 1] - f[i]  # lAter
    A[3][2 * (n - 1) ** 2 + i], A[3][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), - np.exp(mu_u * vertex[1][1])
    A[3][3 * (n - 1) ** 2 + i], A[3][3 * (n - 1) ** 2 + i + 1] = -np.exp(-mu * vertex[1][1]), + np.exp(
        -mu_u * vertex[1][1])
    return A, B

def inter_boundary(vertex, f, a, b, i, n, jd, jv, interface):
    h = 1 / (n - 1)
    ld = int((n-1) * ((interface/h) - 1))
    lu = int((n-1) * ((interface/h) - 1)) + n - 2
    rd = int((n-1) * ((interface/h) - 1)) + n - 1
    ru = int((n-1) * ((interface/h) - 1)) + 2 * n - 3
    mu = np.sqrt(b[i] / a[i])
    mu_l = np.sqrt(b[i - (n - 1)] / a[i - (n - 1)])
    mu_u = np.sqrt(b[i + 1] / a[i + 1])
    if i == ld:
        A = np.zeros((5, 4 * (n - 1) ** 2))
        B = np.zeros((5))
        # left
        A[0][i], A[0][i - (n - 1)] = np.exp(mu * vertex[0][0]), - np.exp(mu_l * vertex[0][0])
        A[0][(n - 1) ** 2 + i], A[0][(n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][0]), - np.exp(
            -mu_l * vertex[0][0])
        A[0][2 * (n - 1) ** 2 + i], A[0][2 * (n - 1) ** 2 + i - (n - 1)] = np.exp(mu * vertex[0][1]), - np.exp(
            mu_l * vertex[0][1])
        A[0][3 * (n - 1) ** 2 + i], A[0][3 * (n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][1]), - np.exp(
            -mu_l * vertex[0][1])
        B[0] = f[i - (n - 1)] - f[i]  # later
        A[1][i], A[1][i - (n - 1)] = -np.exp(mu * vertex[0][0]), np.exp(mu_l * vertex[0][0])
        A[1][(n - 1) ** 2 + i], A[1][(n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][0]), - np.exp(
            -mu_l * vertex[0][0])
        # up
        A[2][i], A[2][i + 1] = np.exp(mu * vertex[1][0]), - np.exp(mu_u * vertex[1][0])
        A[2][(n - 1) ** 2 + i], A[2][(n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][0]), - np.exp(
            -mu_u * vertex[1][0])
        A[2][2 * (n - 1) ** 2 + i], A[2][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), - np.exp(
            mu_u * vertex[1][1])
        A[2][3 * (n - 1) ** 2 + i], A[2][3 * (n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][1]), - np.exp(
            -mu_u * vertex[1][1])
        B[2] = f[i + 1] - f[i]  # lAter
        A[3][2 * (n - 1) ** 2 + i], A[3][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), - np.exp(
            mu_u * vertex[1][1])
        A[3][3 * (n - 1) ** 2 + i], A[3][3 * (n - 1) ** 2 + i + 1] = -np.exp(-mu * vertex[1][1]), + np.exp(
            -mu_u * vertex[1][1])
        #down
        A[4][i] = np.exp(mu * vertex[3][0])
        A[4][(n - 1) ** 2 + i] = np.exp(-mu * vertex[3][0])
        A[4][2 * (n - 1) ** 2 + i] = np.exp(mu * vertex[3][1])
        A[4][3 * (n - 1) ** 2 + i] = np.exp(-mu * vertex[3][1])
        B[4] = -f[i]
        return A, B
    if i == lu:
        A = np.zeros((3, 4 * (n - 1) ** 2))
        B = np.zeros((3))
        # up
        A[0][i] = np.exp(mu * vertex[1][0])
        A[0][(n - 1) ** 2 + i] = np.exp(-mu * vertex[1][0])
        A[0][2 * (n - 1) ** 2 + i] = np.exp(mu * vertex[1][1])
        A[0][3 * (n - 1) ** 2 + i] = np.exp(-mu * vertex[1][1])
        B[0] = -f[i]
        # left
        A[1][i], A[1][i - (n - 1)] = np.exp(mu * vertex[0][0]), - np.exp(mu_l * vertex[0][0])
        A[1][(n - 1) ** 2 + i], A[1][(n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][0]), - np.exp(
            -mu_l * vertex[0][0])
        A[1][2 * (n - 1) ** 2 + i], A[1][2 * (n - 1) ** 2 + i - (n - 1)] = np.exp(mu * vertex[0][1]), - np.exp(
            mu_l * vertex[0][1])
        A[1][3 * (n - 1) ** 2 + i], A[1][3 * (n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][1]), - np.exp(
            -mu_l * vertex[0][1])
        B[1] = f[i - (n - 1)] - f[i]  # lAter
        A[2][i], A[2][i - (n - 1)] = -np.exp(mu * vertex[0][0]), np.exp(mu_l * vertex[0][0])
        A[2][(n - 1) ** 2 + i], A[2][(n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][0]), -np.exp(
            -mu_l * vertex[0][0])
        return A, B
    if i == rd:
        A = np.zeros((5, 4 * (n - 1) ** 2))
        B = np.zeros((5))
        # left
        A[0][i], A[0][i - (n - 1)] = np.exp(mu * vertex[0][0]), - np.exp(mu_l * vertex[0][0])
        A[0][(n - 1) ** 2 + i], A[0][(n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][0]), - np.exp(
            -mu_l * vertex[0][0])
        A[0][2 * (n - 1) ** 2 + i], A[0][2 * (n - 1) ** 2 + i - (n - 1)] = np.exp(mu * vertex[0][1]), - np.exp(
            mu_l * vertex[0][1])
        A[0][3 * (n - 1) ** 2 + i], A[0][3 * (n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][1]), - np.exp(
            -mu_l * vertex[0][1])
        B[0] = f[i - (n - 1)] - f[i] + jv[i]  # lAter
        A[1][i], A[1][i - (n - 1)] = np.exp(mu * vertex[0][0]), - np.exp(mu_l * vertex[0][0])
        A[1][(n - 1) ** 2 + i], A[1][(n - 1) ** 2 + i - (n - 1)] = -np.exp(-mu * vertex[0][0]), np.exp(
            -mu_l * vertex[0][0])
        B[1] = jd[i]
        # up
        A[2][i], A[2][i + 1] = np.exp(mu * vertex[1][0]), - np.exp(mu_u * vertex[1][0])
        A[2][(n - 1) ** 2 + i], A[2][(n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][0]), - np.exp(
            -mu_u * vertex[1][0])
        A[2][2 * (n - 1) ** 2 + i], A[2][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), - np.exp(
            mu_u * vertex[1][1])
        A[2][3 * (n - 1) ** 2 + i], A[2][3 * (n - 1) ** 2 + i + 1] = np.exp(-mu * vertex[1][1]), - np.exp(
            -mu_u * vertex[1][1])
        B[2] = f[i + 1] - f[i]  # lAter
        A[3][2 * (n - 1) ** 2 + i], A[3][2 * (n - 1) ** 2 + i + 1] = np.exp(mu * vertex[1][1]), - np.exp(
            mu_u * vertex[1][1])
        A[3][3 * (n - 1) ** 2 + i], A[3][3 * (n - 1) ** 2 + i + 1] = -np.exp(-mu * vertex[1][1]), + np.exp(
            -mu_u * vertex[1][1])
        # down
        A[4][i] = np.exp(mu * vertex[3][0])
        A[4][(n - 1) ** 2 + i] = np.exp(-mu * vertex[3][0])
        A[4][2 * (n - 1) ** 2 + i] = np.exp(mu * vertex[3][1])
        A[4][3 * (n - 1) ** 2 + i] = np.exp(-mu * vertex[3][1])
        B[4] = -f[i]
        return A, B
    if i == ru:
        A = np.zeros((3, 4 * (n - 1) ** 2))
        B = np.zeros((3))
        # left
        A[0][i], A[0][i - (n - 1)] = np.exp(mu * vertex[0][0]), - np.exp(mu_l * vertex[0][0])
        A[0][(n - 1) ** 2 + i], A[0][(n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][0]), - np.exp(
            -mu_l * vertex[0][0])
        A[0][2 * (n - 1) ** 2 + i], A[0][2 * (n - 1) ** 2 + i - (n - 1)] = np.exp(mu * vertex[0][1]), - np.exp(
            mu_l * vertex[0][1])
        A[0][3 * (n - 1) ** 2 + i], A[0][3 * (n - 1) ** 2 + i - (n - 1)] = np.exp(-mu * vertex[0][1]), - np.exp(
            -mu_l * vertex[0][1])
        B[0] = f[i - (n - 1)] - f[i] + jv[i]  # lAter
        A[1][i], A[1][i - (n - 1)] = np.exp(mu * vertex[0][0]), - np.exp(mu_l * vertex[0][0])
        A[1][(n - 1) ** 2 + i], A[1][(n - 1) ** 2 + i - (n - 1)] = -np.exp(-mu * vertex[0][0]), np.exp(
            -mu_l * vertex[0][0])
        B[1] = jd[i]
        # up
        A[2][i] = np.exp(mu * vertex[1][0])
        A[2][(n - 1) ** 2 + i] = np.exp(-mu * vertex[1][0])
        A[2][2 * (n - 1) ** 2 + i] = np.exp(mu * vertex[1][1])
        A[2][3 * (n - 1) ** 2 + i] = np.exp(-mu * vertex[1][1])
        B[2] = -f[i]
        return A, B


