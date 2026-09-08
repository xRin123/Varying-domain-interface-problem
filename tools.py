import numpy as np
from itertools import product

def get_rectangle_vertices(i, n, x=None):
    """
    获取索引为i的小矩形的四个顶点坐标

    Parameters:
    i: 矩形索引
    n: 网格点数
    x: 网格坐标数组（可选，如果不提供会自动生成）

    Returns:
    vertices: 四个顶点的坐标字典
    position_info: 位置信息字典
    """
    # 计算网格参数
    num_rectangles_per_side = n - 1

    # 检查索引有效性
    if i < 0 or i >= num_rectangles_per_side ** 2:
        raise ValueError(f"索引 {i} 超出范围。有效范围: 0 - {num_rectangles_per_side ** 2 - 1}")

    # 生成或使用提供的网格坐标
    if x is None:
        x = np.linspace(0, 1, num=n)

    # 计算矩形在网格中的位置
    row = i % num_rectangles_per_side  # y方向索引
    col = i // num_rectangles_per_side  # x方向索引

    # 四个顶点的坐标
    bottom_left = (x[col], x[row])
    bottom_right = (x[col + 1], x[row])
    top_left = (x[col], x[row + 1])
    top_right = (x[col + 1], x[row + 1])

    vertices = {
        'bottom_left': bottom_left,
        'bottom_right': bottom_right,
        'top_left': top_left,
        'top_right': top_right
    }

    position_info = {
        'index': i,
        'grid_position': (row, col),
        'row': row,
        'col': col,
        'x_range': (x[col], x[col + 1]),
        'y_range': (x[row], x[row + 1]),
        'width': x[col + 1] - x[col],
        'height': x[row + 1] - x[row]
    }

    return vertices, position_info


def get_rectangle_edge_midpoints(i, n, x=None):
    """
    获取索引为i的小矩形四条边的中点坐标，按照左上右下的顺序

    Parameters:
    i: 矩形索引
    n: 网格点数
    x: 网格坐标数组（可选）

    Returns:
    midpoints: 4x2的numpy数组，按照[上, 左, 右, 下]的顺序
    """
    # 获取矩形的四个顶点坐标
    vertices, position_info = get_rectangle_vertices(i, n, x)

    # 计算四条边的中点坐标
    # 上边中点 (top edge midpoint)
    top_mid = ((vertices['top_left'][0] + vertices['top_right'][0]) / 2,
               vertices['top_left'][1])

    # 左边中点 (left edge midpoint)
    left_mid = (vertices['top_left'][0],
                (vertices['top_left'][1] + vertices['bottom_left'][1]) / 2)

    # 右边中点 (right edge midpoint)
    right_mid = (vertices['top_right'][0],
                 (vertices['top_right'][1] + vertices['bottom_right'][1]) / 2)

    # 下边中点 (bottom edge midpoint)
    bottom_mid = ((vertices['bottom_left'][0] + vertices['bottom_right'][0]) / 2,
                  vertices['bottom_left'][1])

    # 按照左上右下的顺序组成数组
    midpoints = np.array([
        left_mid,  # 左
        top_mid, # 上
        right_mid,  # 右
        bottom_mid  # 下
    ])

    return midpoints


def f(x):
    """被积函数 f(x0, x1) = 4*(cos(x0+x1)+1)*sin(x0+x1)"""
    return 4 * (np.cos(x[0] + x[1]) + 1) * np.sin(x[0] + x[1])


def analytical_mean(a, b):
    """解析解计算的均值"""
    term1 = np.sin(2 * a + 2 * b)
    term2 = -0.5 * np.sin(4 * a) - 0.5 * np.sin(4 * b)
    term3 = 8 * np.sin(a + b)
    term4 = -4 * np.sin(2 * a) - 4 * np.sin(2 * b)

    integral_value = term1 + term2 + term3 + term4
    return integral_value / ((b - a) ** 2)