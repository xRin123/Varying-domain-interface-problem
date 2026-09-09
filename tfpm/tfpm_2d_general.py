import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from subfunc_general import *
from tools import *

#solve interface PDE -a\nabla u + bu = f
#interface: square with l = 0.4
#jump condition: constant jd,jv

def classified_rectangles(n, interface):
    h = 1/(n-1)
    data = {}
    outer_top = [0, n-2, (n-1)*(n-2), n*(n-2)]
    outer_left = [i for i in range(1,n-2)]
    outer_down = [i*(n-1) for i in range(1, n-2)]
    outer_up = [i+n-2 for i in outer_down]
    outer_right = [i + (n-1)*(n-2) for i in outer_left]
    ld = int((n-1) * interface[0][0]/h + interface[0][1]/h)
    inter_top = [ld, ld + int(0.4/h) -1, ld + (n-1) * (int(0.4/h) -1), ld + n * (int(0.4/h) -1)]
    inter_out_left = [ld-(n-1)+i for i in range(int(0.4/h))]
    inter_out_up = [ld+int(0.4/h)+(n-1)*i for i in range(int(0.4/h))]
    inter_out_right = [ld+ (n-1)*int(0.4/h) +i for i in range(int(0.4/h))]
    inter_out_down = [ld-1+(n-1)*i for i in range(int(0.4/h))]
    inter_in_left = [ld + i for i in range(1,int(0.4/h)-1)]
    inter_in_right = [ld + (n-1) * (int(0.4/h) -1) + i for i in range(1,int(0.4/h)-1)]
    inter_in_up = [ld + int(0.4/h)-1 + (n-1)* i for i in range(1,int(0.4/h)-1)]
    inter_in_down = [ld + (n-1)* i for i in range(1,int(0.4/h)-1)]
    data['outer_top'] = outer_top
    data['outer_left'] = outer_left
    data['outer_up'] = outer_up
    data['outer_right'] = outer_right
    data['outer_down'] = outer_down
    data['inter_top'] = inter_top
    data['inter_out_left'] = inter_out_left
    data['inter_out_up'] = inter_out_up
    data['inter_out_right'] = inter_out_right
    data['inter_out_down'] = inter_out_down
    data['inter_in_left'] = inter_in_left
    data['inter_in_up'] = inter_in_up
    data['inter_in_right'] = inter_in_right
    data['inter_in_down'] = inter_in_down
    indexes = outer_top + outer_left + outer_right + outer_up + outer_down + inter_top + inter_out_left + inter_out_up + inter_out_right + inter_out_down \
                    + inter_in_left + inter_in_up + inter_in_right + inter_in_down
    return data, indexes

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

def tfpm_2d(F, a, b, jd, jv, n, interface):
    # n: number of uniform mesh on one dim
    # a, b, f: functions
    # interface: polygon interface m*2
    # 4 * n**2 coefficient: ci0,ci1,ci2,ci3 for n**2 meshes
    data, index = classified_rectangles(n, interface)
    A = []
    B = []
    for i in range((n-1)**2):
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
    #A = np.vstack(A)
    #c = np.linalg.matrix_rank(A)
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
    for i in data['inter_top']:
        vertex = get_rectangle_edge_midpoints(i, n)
        ai, bi = inter_top(vertex, F, a, b, i, n, jd, jv, interface)
        A.append(ai)
        B.append(bi)
    for i in data['inter_out_left']:
        vertex = get_rectangle_edge_midpoints(i, n)
        ai, bi = inter_out_left(vertex, F, a, b, i, n, jd, jv)
        A.append(ai)
        B.append(bi)
    for i in data['inter_out_up']:
        vertex = get_rectangle_edge_midpoints(i, n)
        ai, bi = inter_out_up(vertex, F, a, b, i, n, jd, jv)
        A.append(ai)
        B.append(bi)
    for i in data['inter_out_right']:
        vertex = get_rectangle_edge_midpoints(i, n)
        ai, bi = inter_out_right(vertex, F, a, b, i, n, jd, jv)
        A.append(ai)
        B.append(bi)
    for i in data['inter_out_down']:
        vertex = get_rectangle_edge_midpoints(i, n)
        ai, bi = inter_out_down(vertex, F, a, b, i, n, jd, jv)
        A.append(ai)
        B.append(bi)
    for i in data['inter_in_left']:
        vertex = get_rectangle_edge_midpoints(i, n)
        ai, bi = inter_in_left(vertex, F, a, b, i, n, jd, jv)
        A.append(ai)
        B.append(bi)
    for i in data['inter_in_up']:
        vertex = get_rectangle_edge_midpoints(i, n)
        ai, bi = inter_in_up(vertex, F, a, b, i, n, jd, jv)
        A.append(ai)
        B.append(bi)
    for i in data['inter_in_right']:
        vertex = get_rectangle_edge_midpoints(i, n)
        ai, bi = inter_in_right(vertex, F, a, b, i, n, jd, jv)
        A.append(ai)
        B.append(bi)
    for i in data['inter_in_down']:
        vertex = get_rectangle_edge_midpoints(i, n)
        ai, bi = inter_in_down(vertex, F, a, b, i, n, jd, jv)
        A.append(ai)
        B.append(bi)
    A = np.vstack(A)
    B = np.hstack(B)
    c = np.linalg.solve(A,B)
    return c

def main():
    u = []
    i = np.random.randint(5000)
    n = 51
    a = np.load('../data/square/general/a.npy')[i].reshape(-1)
    a = a + np.abs(np.min(a)) + 0.5
    b = np.load('../data/square/general/b.npy')[i].reshape(-1)
    b = b + np.abs(np.min(b)) + 0.5
    F = np.load('../data/square/general/f.npy')[i].reshape(-1)/b
    jd = 0.1 * np.load('../data/square/general/jd.npy')[i] * np.ones(((n-1)**2))/a
    jv = 0.1 * np.load('../data/square/general/jv.npy')[i] * np.ones(((n-1)**2))
    interface = np.array([[0.3,0.3],[0.3,0.7],[0.7,0.7],[0.7,0.3]])
    interface_plot = np.concatenate((interface, interface[0].reshape(1,-1)),axis=0)
    c = tfpm_2d(F, a, b, jd, jv, n, interface)
    x = np.linspace(0.005, 0.995, n-1)
    points = np.array(list(product(x, x)))

    uk = []
    for j in range((n-1)**2):
        mu = np.sqrt(b[j] / a[j])
        uj = F[j] + c[j]*np.exp(mu*points[j][0]) + c[j + (n-1)**2]*np.exp(-mu*points[j][0]) + c[j + 2*(n-1)**2]*np.exp(mu*points[j][1]) + c[j+3*(n-1)**2]*np.exp(-mu*points[j][1])
        uk.append(uj)
    uk = np.array(uk).reshape(n-1,n-1)
    u.append(uk)

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    lims = dict(cmap='rainbow')
    ax[0][0].set_title('Solution')
    tpc0 = ax[0][0].scatter(points[..., 0], points[..., 1], 100, uk, edgecolor='w', lw=0.1, **lims)
    tpc1 = ax[0][1].scatter(points[..., 0], points[..., 1], 100, c[:(n-1)**2], edgecolor='w', lw=0.1, **lims)
    tpc2 = ax[0][2].scatter(points[..., 0], points[..., 1], 100, c[(n-1)**2:2*(n - 1) ** 2], edgecolor='w', lw=0.1, **lims)
    tpc3 = ax[1][1].scatter(points[..., 0], points[..., 1], 100, c[2*(n-1)**2:3*(n - 1) ** 2], edgecolor='w', lw=0.1, **lims)
    tpc4 = ax[1][2].scatter(points[..., 0], points[..., 1], 100, c[3 * (n - 1) ** 2:], edgecolor='w',
                                lw=0.1, **lims)
    ax[1][0].set_title(r'$f$')
    tpc5 = ax[1][0].scatter(points[..., 0], points[..., 1], 100, F, edgecolor='w',
                                lw=0.1, **lims)
    ax[0][0].plot(interface_plot[:,0], interface_plot[:,1])
    fig.colorbar(tpc0)
    fig.colorbar(tpc1)
    fig.colorbar(tpc2)
    fig.colorbar(tpc3)
    fig.colorbar(tpc4)
    fig.colorbar(tpc5)
    plt.show()

if __name__ == '__main__':
    main()