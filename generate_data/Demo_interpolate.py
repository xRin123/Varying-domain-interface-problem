import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
import datetime
from generatedata.generate_inner_boundary import get_barycenter, get_boundary_pts_theta
from scipy.interpolate import griddata
from itertools import product


def fun_inter(pts, func, n):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)

    interpolator = RegularGridInterpolator(
        (x, y),  # 网格坐标（x 方向和 y 方向）
        func,  # 函数值，形状 (n, n)
        method='cubic'  # 三次插值，可选 'linear', 'nearest'
    )

    value = interpolator(pts)

    return value


def fun_inter_3d(pts, func, n):

    # 创建均匀网格坐标
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    z = np.linspace(0, 1, n)

    # 创建三维插值器
    # RegularGridInterpolator 接受 (x, y, z) 作为网格，func 作为值
    interpolator = RegularGridInterpolator(
        (x, y, z),  # 网格坐标（元组形式）
        func,  # 三维数组，形状 (n, n, n)
        method='cubic'  # 三次插值（可选 'linear', 'nearest'）
    )

    # 对输入点进行插值
    # pts 需要是 (N, 3) 形状，每一行是 (x, y, z)
    value = interpolator(pts)

    return value

def fun_inter_boundary(pts, poly, func):
    x = np.linspace(0, 2*np.pi, func.shape[0])
    pts = pts - get_barycenter(poly)
    pts_r = np.linalg.norm(pts, axis=1, keepdims=True)
    pts_sin = pts[:, 1:2] / pts_r
    sign = np.sign(pts[:, 0:1])
    bias = np.pi * (1 - sign) / 2
    theta = (sign * np.arcsin(pts_sin) + bias) % (2 * np.pi)
    f = interpolate.interp1d(x, func, kind='cubic')
    return np.array(f(theta)).reshape(-1)

def fun_inter_g_boundary_sample(func):
    x = np.linspace(0, 2 * np.pi, func.shape[0])
    t = np.linspace(0, 2 * np.pi, 200)
    f = interpolate.interp1d(x, func, kind='cubic')
    return np.array(f(t)).reshape(-1)

def func_inter_3d(func, pts):
    x = np.linspace(0,1,20)
    points = np.array(list(product(x, x, x)))
    value = func.reshape(-1, 8000)
    f_inter = []
    for i in range(func.shape[0]):
        print('caculate {}_th'.format(i))
        fi = griddata(points, value[i], pts, method='linear')
        f_inter.append(fi)
    f_inter = np.array(f_inter).reshape(func.shape[0], -1)
    return f_inter

def deter_outer_or_inner(inner, outer, pts):
    center = get_barycenter(inner)
    pts = pts - center
    b_r = np.linalg.norm(pts, axis=1, keepdims=True)
    b_sin = pts[:, 1:2] / b_r
    sign = np.sign(pts[:, 0:1])
    bias = np.pi * (1 - sign) / 2
    theta = (sign * np.arcsin(b_sin) + bias) % (2 * np.pi)
    theta[b_sin == -1] = 3 / 2 * np.pi
    inner_r, _ = get_boundary_pts_theta(inner, theta)
    outer_r, _ = get_boundary_pts_annular_theta(inner, outer, theta)
    d = np.concatenate(((inner_r - b_r) ** 2, (outer_r - b_r) ** 2), axis=-1)
    error_index = np.where(d > 10000)
    d[error_index] = 0
    index = np.argpartition(d, kth=1, axis=1)[:, 0]
    inner_index = np.where(index == 0)
    outer_index = np.where(index == 1)
    return pts[outer_index] + center, pts[inner_index] + center

def xy_2_r(pts, poly):
    center = get_barycenter(poly)
    pts = pts - center
    pts_r = np.linalg.norm(pts, axis=1, keepdims=True)
    pts_sin = pts[:, 1:2] / pts_r
    sign = np.sign(pts[:, 0:1])
    bias = np.pi * (1 - sign) / 2
    theta = (sign * np.arcsin(pts_sin) + bias) % (2 * np.pi)
    theta[pts_sin == -1] = 3 / 2 * np.pi
    r = get_boundary_pts_theta(poly - center, theta)[0]
    return theta, r

def main():
    areas = np.load('../data/polar_line/area_sample_xy.npy')
    u_uniform = np.load('../data/polar_line/solve_uniform.npy')
    u_uniform_nonc = np.load('../data/polar_line_nonc/solve_uniform.npy')
    mesh = np.load('../data/polar_line/meshes_test.npz')
    s_c = {}
    s_nonc = {}
    for i in range(3000, 4000):
        center = get_barycenter(areas[i])
        s1 = fun_inter(mesh['points_{}'.format(i)] - center + np.array([0.5,0.5]), u_uniform[i], 100)
        s2 = fun_inter(mesh['points_{}'.format(i)] - center + np.array([0.5,0.5]), u_uniform_nonc[i], 100)
        s_c['solve_{}'.format(i)] = s1
        s_nonc['solve_{}'.format(i)] = s2
    np.savez_compressed('../data/polar_line/solve_mesh_test.npz', **s_c)
    np.savez_compressed('../data/polar_line_nonc/solve_mesh_test.npz', **s_nonc)

if __name__ == '__main__':
    main()