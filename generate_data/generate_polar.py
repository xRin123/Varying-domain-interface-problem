import matplotlib.pyplot as plt
import numpy as np
import pygmsh
from itertools import product
from PDE.generatedata.gaussian_process import Gaussian_process_period
from Ray_Casting_Algorithm import is_point_in_polygon
from shapely.geometry import Polygon, Point

def radius_2_space(r):

    theta = np.linspace(0, 2 * np.pi, r.shape[-1], endpoint=False)
    x = r * (np.cos(theta)) + 0.5
    y = r * (np.sin(theta)) + 0.5
    points = np.concatenate((x[...,None], y[...,None]), axis=-1)
    return points

def generate_area(num):
    intervals = [0, 1]
    mean = 1
    std = 0.2
    length_scale = 0.6 #0.5
    features = 1001
    period = 1

    gp = Gaussian_process_period(intervals, mean, std, length_scale, features, period)
    gps = gp.generate(num)
    theta = np.linspace(0, 2 * np.pi, 1001)
    x = gps * (np.cos(theta))
    y = gps * (np.sin(theta))
    scale = 2 * max(np.max(np.abs(x)), np.max(np.abs(y)))
    x = x/scale + 0.5
    y = y/scale + 0.5
    points = np.concatenate((x.reshape(num, features, 1), y.reshape(num, features, 1)), axis=-1)
    return points#(gps/scale)[:,:1000]

def generate_chara(areas, min, max, n = 100):
    x = np.linspace(min, max, num=n)
    points = np.array(list(product(x, x))).reshape(n, n, 2)
    chara_matrix = []
    for l in range(areas.shape[0]):
        matrix = np.zeros((n,n))
        areal = areas[l]
        for i in range(n):
            for j in range(n):
                if is_point_in_polygon(points[i][j], areal):
                    if points[i][j][0]>=0:
                        matrix[i][j] = 2
                    else:
                        matrix[i][j] = 1
        chara_matrix.append(matrix)
    chara_matrix = np.array(chara_matrix).reshape(areas.shape[0], n, n)
    return chara_matrix

def generate_uniform_pts(n = 100, min = -0.5, max = 0.5):
    x = np.linspace(min, max, num=n)
    points = np.array(list(product(x, x))).reshape(n, n, 2)
    return points

def generate_square():
    num = 3000

    l = 0.2
    x = 0.01 * np.random.randint(30, 70, 3000).reshape(-1,1)
    y = 0.01 * np.random.randint(30, 70, 3000).reshape(-1,1)

    center = np.concatenate((x,y), axis=-1)
    x00 = np.concatenate((x-l,y-l), axis=-1).reshape(-1, 1, 2)
    x01 = np.concatenate((x-l,y+l), axis=-1).reshape(-1, 1, 2)
    x11 = np.concatenate((x+l,y+l), axis=-1).reshape(-1, 1, 2)
    x10 = np.concatenate((x+l,y-l), axis=-1).reshape(-1, 1, 2)
    area = np.concatenate((x00, x01, x11, x10), axis=1)

    return center, area

def main():

    areas = np.load('../data/square/area_xy.npy')

if __name__ == '__main__':
    main()