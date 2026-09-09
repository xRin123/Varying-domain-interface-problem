import matplotlib.pyplot as plt
import numpy as np
from tfpm.tools import *
from Ray_Casting_Algorithm import *
from gaussian_process import *

def gen_mesh_uniform_3d(n):
    x = np.linspace(0, 1, num=n+1)
    points = np.array(list(product(x, x, x)))
    index_pt = np.arange((n+1) ** 3).reshape(n+1, n+1, n+1)
    cube_rd_0 = index_pt[:n][:,:n][:,:,:n].reshape(-1,1)
    cube_0 = np.concatenate((cube_rd_0, cube_rd_0+1, cube_rd_0+n+2, cube_rd_0 + n+1, cube_rd_0 + (n+1)**2, \
                             cube_rd_0 + (n+1)**2 + 1, cube_rd_0 + (n+1)**2 + n+2, cube_rd_0 + (n+1)**2+n+1), axis=-1)
    tetra1 = np.concatenate((cube_0[:,4].reshape(-1,1), cube_0[:,0].reshape(-1,1), cube_0[:,1].reshape(-1,1), cube_0[:,3].reshape(-1,1)), axis=-1)
    tetra2 = np.concatenate((cube_0[:,4].reshape(-1,1), cube_0[:,5].reshape(-1,1), cube_0[:,1].reshape(-1,1), cube_0[:,6].reshape(-1,1)), axis=-1)
    tetra3 = np.concatenate((cube_0[:, 2].reshape(-1, 1), cube_0[:, 1].reshape(-1, 1), cube_0[:, 6].reshape(-1, 1),
                             cube_0[:, 3].reshape(-1, 1)), axis=-1)
    tetra4 = np.concatenate((cube_0[:, 1].reshape(-1, 1), cube_0[:, 4].reshape(-1, 1), cube_0[:, 6].reshape(-1, 1),
                             cube_0[:, 3].reshape(-1, 1)), axis=-1)
    tetra5 = np.concatenate((cube_0[:, 7].reshape(-1, 1), cube_0[:, 3].reshape(-1, 1), cube_0[:, 6].reshape(-1, 1),
                             cube_0[:, 4].reshape(-1, 1)), axis=-1)
    tetras = np.concatenate((tetra1, tetra2, tetra3, tetra4, tetra5), axis=0)
    vertex = np.array([[0,0,0],[0,1,0],[0,0,1],[0,1,1],
                       [1,0,0],[1,1,0],[1,0,1],[1,1,1]])
    cond1 = points[:,0] == 0
    cond2 = points[:,1] == 0
    cond3 = points[:,2] == 0
    cond4 = points[:,0] == 1
    cond5 = points[:,1] == 1
    cond6 = points[:,2] == 1
    b_index = np.hstack((np.array(np.where(cond1)), np.array(np.where(cond2)), np.array(np.where(cond3)),\
                         np.array(np.where(cond4)), np.array(np.where(cond5)), np.array(np.where(cond6)))).reshape(-1)
    b_index = list(set(b_index.tolist()))
    b_index = np.array(b_index).reshape(-1)
    mesh = {
        'points': points,
        'vertex': vertex,
        'tetra' : tetras,
        'boundary': b_index
    }
    return mesh

def generate_square_3d(n, step):
    start = 0.3
    end = 0.7
    l = 0.4

    grid_points = np.arange(start, end + step / 2, step)
    x = np.random.choice(grid_points, size=n, replace=True)
    y = np.random.choice(grid_points, size=n, replace=True)
    x0 = np.concatenate(((x-l/2).reshape(-1,1), (y-l/2).reshape(-1,1)),axis=-1)
    x1 = np.concatenate(((x-l/2).reshape(-1,1), (y+l/2).reshape(-1,1)),axis=-1)
    x2 = np.concatenate(((x + l / 2).reshape(-1, 1), (y + l / 2).reshape(-1, 1)), axis=-1)
    x3 = np.concatenate(((x + l / 2).reshape(-1, 1), (y - l / 2).reshape(-1, 1)), axis=-1)
    area = np.concatenate((x0.reshape(-1,1,2),x1.reshape(-1,1,2),x2.reshape(-1,1,2),x3.reshape(-1,1,2)), axis=1)
    return area

def plot_3d_surface(amp, freq_x, freq_y, phi_x, phi_y, n = 100):
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)

    Z = 0.5 + amp * np.sin(2 * np.pi * freq_x * X + phi_x) * \
        np.sin(2 * np.pi * freq_y * Y + phi_y)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap='viridis',
                           rstride=4, cstride=4, alpha=0.9,
                           linewidth=0.5, edgecolors='k', antialiased=True)

    cube_color = 'black'
    for y_e in [0, 1]:
        for z_e in [0, 1]:
            ax.plot([0, 1], [y_e, y_e], [z_e, z_e], color=cube_color, lw=1)
    for x_e in [0, 1]:
        for z_e in [0, 1]:
            ax.plot([x_e, x_e], [0, 1], [z_e, z_e], color=cube_color, lw=1)
    for x_e in [0, 1]:
        for y_e in [0, 1]:
            ax.plot([x_e, x_e], [y_e, y_e], [0, 1], color=cube_color, lw=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=30, azim=-60)

    plt.title("Surface $Z = 0.5 + 0.2 \sin(2\pi X) \sin(2\pi Y)$")
    plt.show()
