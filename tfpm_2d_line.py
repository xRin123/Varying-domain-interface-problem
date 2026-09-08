import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from tools import *
from subfunc_general_line import *

#solve interface PDE -a\nabla u + bu = f
#interface: x = c
#jump condition: constant jd,jv

def classified_rectangles(n, interface):
    h = 1/(n-1)
    ld = int((n-1) * ((interface/h) - 1))
    data = {}
    outer_top = [0, n-2, (n-1)*(n-2), n*(n-2)]
    outer_left = [i for i in range(1,n-2)]
    outer_down = [i*(n-1) for i in range(1, n-2)]
    outer_up = [i+n-2 for i in outer_down]
    outer_right = [i + (n-1)*(n-2) for i in outer_left]
    inter_right = [int((n-1) * ((interface/h) - 1)) + n - 1 + i for i in range(1, n-2)]
    inter_boundary = [ld, ld + n-1, ld + n-2, ld + 2*n -3]
    outer_up.remove(int((n-1) * ((interface/h) - 1)) + n - 2)
    outer_up.remove(int((n - 1) * ((interface / h) - 1)) + 2 * n - 3)
    outer_down.remove(int((n-1) * ((interface/h) - 1)))
    outer_down.remove(int((n-1) * ((interface/h) - 1)) + n - 1)
    data['outer_top'] = outer_top
    data['outer_left'] = outer_left
    data['outer_up'] = outer_up
    data['outer_right'] = outer_right
    data['outer_down'] = outer_down
    data['inter_right'] = inter_right
    data['inter_boundary'] = inter_boundary
    indexes = outer_top + outer_left + outer_right + outer_up + outer_down + inter_right + inter_boundary
    return data, indexes

def tfpm_2d(F, a, b, jd, jv, n, interface):
    # n: number of uniform mesh on one dim
    # a, b, f: functions
    # interface: constant
    # 4 * n**2 coefficient: ci0,ci1,ci2,ci3 for n**2 meshes
    data, index = classified_rectangles(n, interface)
    A = []
    B = []
    for i in range((n - 1) ** 2):
        if i not in index:
            vertex = get_rectangle_edge_midpoints(i,n)
            ai, bi = normal(vertex, F, a, b, i, n)
            A.append(ai)
            B.append(bi)
    for i in data['outer_top']:
        vertex = get_rectangle_edge_midpoints(i, n)
        ai, bi = top(vertex, F, a, b, i, n)
        A.append(ai)
        B.append(bi)
    for i in data['outer_left']:
        vertex = get_rectangle_edge_midpoints(i, n)
        ai, bi = left_outer(vertex, F, a, b, i, n)
        A.append(ai)
        B.append(bi)
    for i in data['outer_up']:
        vertex = get_rectangle_edge_midpoints(i, n)
        ai, bi = up_outer(vertex, F, a, b, i, n)
        A.append(ai)
        B.append(bi)
    # A = np.vstack(A)
    # c = np.linalg.matrix_rank(A)
    for i in data['outer_right']:
        vertex = get_rectangle_edge_midpoints(i, n)
        ai, bi = right_outer(vertex, F, a, b, i, n)
        A.append(ai)
        B.append(bi)
    for i in data['outer_down']:
        vertex = get_rectangle_edge_midpoints(i, n)
        ai, bi = down_outer(vertex, F, a, b, i, n)
        A.append(ai)
        B.append(bi)
    for i in data['inter_boundary']:
        vertex = get_rectangle_edge_midpoints(i, n)
        ai, bi = inter_boundary(vertex, F, a, b, i, n, jd, jv, interface)
        A.append(ai)
        B.append(bi)
    for i in data['inter_right']:
        vertex = get_rectangle_edge_midpoints(i, n)
        ai, bi = inter_right(vertex, F, a, b, i, n, jd, jv)
        A.append(ai)
        B.append(bi)
    A = np.vstack(A)
    B = np.hstack(B)
    c = np.linalg.solve(A, B)
    return c


def main():
    n = 51
    inter = 0.5
    i = np.random.randint(2000)
    a = np.ones(((n-1)**2))
    b = np.ones(((n-1)**2))
    F = np.ones(((n-1)**2)) #np.load('../data/square/general/f.npy')[0]
    jd = 0.2 * np.ones(((n-1)**2))
    jv = 0.2 * np.ones(((n - 1) ** 2))
    c = tfpm_2d(F, a, b, jd, jv, n, inter)
    #x = np.linspace(0, 1, 51)
    x = np.linspace(0.005, 0.995, n - 1)
    points = np.array(list(product(x, x)))

    uk = []
    for j in range((n - 1) ** 2):
        mu = np.sqrt(b[j] / a[j])
        uj = F[j] + c[j] * np.exp(mu * points[j][0]) + c[j + (n - 1) ** 2] * np.exp(-mu * points[j][0]) + c[
            j + 2 * (n - 1) ** 2] * np.exp(mu * points[j][1]) + c[j + 3 * (n - 1) ** 2] * np.exp(-mu * points[j][1])
        uk.append(uj)
    uk = np.array(uk).reshape(n - 1, n - 1)


    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    lims = dict(cmap='rainbow')
    ax[0][0].set_title('Solution')
    tpc0 = ax[0][0].scatter(points[..., 0], points[..., 1], 100, uk, edgecolor='w', lw=0.1, **lims)
    tpc1 = ax[0][1].scatter(points[..., 0], points[..., 1], 100, c[:(n - 1) ** 2], edgecolor='w', lw=0.1, **lims)
    tpc2 = ax[0][2].scatter(points[..., 0], points[..., 1], 100, c[(n - 1) ** 2:2 * (n - 1) ** 2], edgecolor='w',
                            lw=0.1, **lims)
    tpc3 = ax[1][1].scatter(points[..., 0], points[..., 1], 100, c[2 * (n - 1) ** 2:3 * (n - 1) ** 2], edgecolor='w',
                            lw=0.1, **lims)
    tpc4 = ax[1][2].scatter(points[..., 0], points[..., 1], 100, c[3 * (n - 1) ** 2:], edgecolor='w',
                            lw=0.1, **lims)
    ax[1][0].set_title(r'$f$')
    tpc5 = ax[1][0].scatter(points[..., 0], points[..., 1], 100, F, edgecolor='w',
                            lw=0.1, **lims)
    fig.colorbar(tpc0)
    fig.colorbar(tpc1)
    fig.colorbar(tpc2)
    fig.colorbar(tpc3)
    fig.colorbar(tpc4)
    fig.colorbar(tpc5)
    # plt.savefig("tfpm_hf.pdf")
    plt.show()

if __name__ == '__main__':
    main()