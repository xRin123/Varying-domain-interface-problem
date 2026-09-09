import matplotlib.pyplot as plt
import numpy as np
import matplotlib.path as mplPath
from shapely.geometry import Point, Polygon
from itertools import product
from Ray_Casting_Algorithm import is_point_in_polygon

def point_to_polygon_distance_shapely(points, polygon):
    """
    使用shapely计算点到多边形距离
    """
    # 创建多边形对象
    poly = Polygon(polygon)
    boundary = poly.boundary

    distances = np.array([Point(p).distance(boundary) for p in points])
    return distances

def generate_sdf_area(interface, n):
    x = np.linspace(0, 1, n)
    points = np.array(list(product(x, x)))
    sdf_area = np.zeros((n ** 2))
    for i in range(n**2):
        d = point_to_polygon_distance_shapely(points[i].reshape(1,2), interface)
        if is_point_in_polygon(points[i], interface):
            sdf_area[i] = -d
        else:
            sdf_area[i] = d
    return sdf_area

def generate_sdf_3d(z, n):
    x = np.linspace(0, 1, n)
    points = np.array(list(product(x, x, x)))
    sdf_area = np.zeros(n**3)
    for i in range(n ** 3):
        pti = points[i]
        sdf_area[i] = pti[-1] - z
    return sdf_area

def surface(freqx, freqy, phix, phiy, x):
    X = x[:, 0]
    Y = x[:, 1]

    Z = 0.5 + 0.2 * np.sin(2 * np.pi * freqx * X + phix) * \
        np.sin(2 * np.pi * freqy * Y + phiy)
    return Z

def generate_sdf_surface(freqx, freqy, phix, phiy, n):
    x = np.linspace(0, 1, n)
    points = np.array(list(product(x, x, x)))
    sdf_area = np.zeros(n ** 3)
    z = surface(freqx, freqy, phix, phiy, points)
    for i in range(n ** 3):
        pti = points[i]
        sdf_area[i] = pti[-1] - z[i]
    return sdf_area

def direct_gaussian_sdf(sdf, constant_value, n):
    """
    方法1: 直接在SDF上应用高斯核
    u(x) = c * exp(-sdf(x)² / (2σ²))
    """
    # 高斯核
    dx = 1.0 / (n - 1)
    sigma = 1.0 * dx
    u_gaussian = constant_value * np.exp(-sdf**2 / (2 * sigma**2))
    u_gaussian[u_gaussian < 1e-8] = 0
    return u_gaussian

def main():
    inter = np.load('../data/3d/area_inter.npy')
    i = 0
    sdf = generate_sdf_3d(inter[i], 26).reshape(1,26,26,26)
    np.save('../data/3d_surface/plane_test.npy', sdf)

if __name__ == '__main__':
    main()
