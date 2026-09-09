import matplotlib.pyplot as plt
import numpy as np
from tfpm.tools import *
from Ray_Casting_Algorithm import *


def interface_index(n, interface):
    h = 1 / (n - 1)
    ld = int((n - 1) * interface[0][0] / h + interface[0][1] / h)
    inter_top = [ld, ld + int(0.4 / h) - 1, ld + (n - 1) * (int(0.4 / h) - 1), ld + n * (int(0.4 / h) - 1)]
    index = []
    for i in range(int(0.4/h)):
        for j in range(int(0.4/h)):
            x = ld + i + j*(n-1)
            index.append(x)
    return index

def generate_square(n, step):
    start = 0.3
    end = 0.7
    l = 0.4

    grid_points = np.arange(start, end, step)
    x = np.random.choice(grid_points, size=n, replace=True)
    y = np.random.choice(grid_points, size=n, replace=True)
    x0 = np.concatenate(((x-l/2).reshape(-1,1), (y-l/2).reshape(-1,1)),axis=-1)
    x1 = np.concatenate(((x-l/2).reshape(-1,1), (y+l/2).reshape(-1,1)),axis=-1)
    x2 = np.concatenate(((x + l / 2).reshape(-1, 1), (y + l / 2).reshape(-1, 1)), axis=-1)
    x3 = np.concatenate(((x + l / 2).reshape(-1, 1), (y - l / 2).reshape(-1, 1)), axis=-1)
    area = np.concatenate((x0.reshape(-1,1,2),x1.reshape(-1,1,2),x2.reshape(-1,1,2),x3.reshape(-1,1,2)), axis=1)
    return area


def generate_F(n, interface):
    F = []
    for i in range(interface.shape[0]):
        Fi = np.ones(((n-1)**2))
        inter = interface_index(n, interface[i])
        for j in range((n - 1) ** 2):
            if j in inter:
                Fi[j] = 0.2
        F.append(Fi)
    F = np.array(F)
    return F

def generate_c_train(n, c):
    c_train = np.zeros((n**2, 4))
    h = 1 / 50
    x = np.linspace(0, 1, 101)
    points = np.array(list(product(x, x)))
    for i in range(points.shape[0]):
        a,b = int(points[i][0]/h), int(points[i][1]/h)
        if a == 50:
            a = 49
        if b == 50:
            b = 49
        j = a*50 + b
        c_train[i][0] = c[j]
        c_train[i][1] = c[2500 + j]
        c_train[i][2] = c[5000 + j]
        c_train[i][3] = c[7500 + j]
    return c_train

def solve_mesh(F, c, a, b, n, m):
    #c: coeffients on square
    h = 1/(n-1)
    x = np.linspace(0, 1, m)
    points = np.array(list(product(x, x)))
    u = []
    F = F/b
    for i in range(points.shape[0]):
        s, t = int(points[i][0]/h), int(points[i][1]/h)
        if s == 50:
            s = 49
        if t == 50:
            t = 49
        j = s*50 + t
        mu = np.sqrt(b[j] / a[j])
        ui = F[j] + c[j]*np.exp(mu*points[i][0])+ c[j + (n-1)**2]*np.exp(-mu*points[i][0]) + \
                 c[j + 2*(n-1)**2]*np.exp(mu*points[i][1]) + c[j+3*(n-1)**2]*np.exp(-mu*points[i][1])
        u.append(ui)
    u = np.array(u)
    return u

def generate_chara_area(interface, n):
    x = np.linspace(0, 1, n)
    points = np.array(list(product(x, x)))
    chara_area = np.zeros((n ** 2))
    for i in range(n**2):
        if is_point_in_polygon(points[i], interface):
            chara_area[i] = 1
    return chara_area

def generate_chara_area2(interface):
    x = np.linspace(0.005, 0.995, 50)
    points = np.array(list(product(x, x)))
    chara_area = np.zeros((50 ** 2))
    for i in range(50**2):
        if is_point_in_polygon(points[i], interface):
            chara_area[i] = 1
    return chara_area

def generate_interface_chara(interface, n):
    x = np.linspace(0, 1, n)
    points = np.array(list(product(x, x)))
    chara_inter = np.zeros((n ** 2))
    for i in range(n**2):
        point = points[i].reshape(1,2)
        if point_to_polygon_distance_shapely(point, interface) < 1e-5:
            chara_inter[i] = 1
    return chara_inter

def generate_hf_f(n):
    m = 51
    x = np.linspace(0.005, 0.995, m - 1)
    points = np.array(list(product(x, x)))
    points_u = np.load('../data/square/points_uniform.npy').reshape(-1, 2)

    #a = (5*np.pi)**2 + (10*np.pi)**2

    kx_sample = np.random.uniform(7 * np.pi, 12 * np.pi, n)
    ky_sample = np.random.uniform(7 * np.pi, 12 * np.pi, n)

    epsilon = 1  # 小振幅
    b = 1.0  # 常数系数

    f = []
    f_train = []

    for i in range(n):
        kx = kx_sample[i]
        ky = ky_sample[i]

        def f_source(x):
            return (kx ** 2 + ky ** 2 + b) * epsilon * np.cos(kx * x[0]) * np.cos(ky * x[1])

        u = []
        u_train = []
        for i in range(points.shape[0]):
            u.append(f_source(points[i]))
        for i in range(points_u.shape[0]):
            u_train.append(f_source(points_u[i]))
        u = np.array(u)
        u_train = np.array(u_train)
        f.append(u)
        f_train.append(u_train)

    f = np.array(f).reshape(n, m - 1, m - 1)
    f_train = np.array(f_train).reshape(n, 101, 101)
    return f, f_train, kx_sample, ky_sample

def tfpm_c_2_u(a, b, F, c, num):
    n = 51
    x = np.linspace(0.005, 0.995, n - 1)
    points = np.array(list(product(x, x)))
    solve = []
    for j in range(num):
        Fj = F[j]
        cj = c[j]
        aj = a[j]
        bj = b[j]
        aj = aj + np.abs(np.min(aj)) + 0.5
        bj = bj + np.abs(np.min(bj)) + 0.5
        Fj = Fj / bj
        u = []
        for i in range((n - 1) ** 2):
            mu = np.sqrt(bj[i] / aj[i])
            ui = Fj[i] + cj[i] * np.exp(mu * points[i][0]) + cj[i + (n - 1) ** 2] * np.exp(-mu * points[i][0]) + cj[
                i + 2 * (n - 1) ** 2] * np.exp(mu * points[i][1]) + cj[i + 3 * (n - 1) ** 2] * np.exp(-mu * points[i][1])
            u.append(ui)
        u = np.array(u).reshape(n - 1, n - 1)
        solve.append(u)
    return np.array(solve)

def tfpm_c_2_u_simple(a, F, c, num):
    n = 51
    x = np.linspace(0.005, 0.995, n - 1)
    points = np.array(list(product(x, x)))
    solve = []
    for j in range(num):
        cj = c[j]
        u = []
        for i in range((n - 1) ** 2):
            mu = np.sqrt(1/a)
            ui = F[i] + cj[i] * np.exp(mu * points[i][0]) + cj[i + (n - 1) ** 2] * np.exp(-mu * points[i][0]) + cj[
                i + 2 * (n - 1) ** 2] * np.exp(mu * points[i][1]) + cj[i + 3 * (n - 1) ** 2] * np.exp(
                -mu * points[i][1])
            u.append(ui)
        u = np.array(u).reshape(n - 1, n - 1)
        solve.append(u)
    return np.array(solve)

def generate_line(num):

    grid_points = np.arange(0.1, 0.9, 0.05)
    line = np.random.choice(grid_points, size=num, replace=True)
    return  line


def main():
    n = 101
    num = 2000
    line = generate_line(num)
    np.save('../data/square_line/line', line)


if __name__ == '__main__':
    main()