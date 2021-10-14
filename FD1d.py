import numpy as np


def init_FD1d_grid(x_min: float, x_max: float, N: int):
    x = np.linspace(x_min, x_max, N)
    h = x[1] - x[0]
    return x, h


def build_D2_matrix_3pt(N: int, h: float):
    mat = np.diag(np.ones(N-1), 1) + \
          np.diag(np.ones(N-1), -1) + \
          np.diag(np.ones(N)) * (-2)
    return mat / h**2


def build_D2_matrix_5pt(N: int, h: float):
    mat = np.diag(np.ones(N-1), 1) * 16 + \
          np.diag(np.ones(N-1), -1) * 16 + \
          np.diag(np.ones(N)) * (-30) + \
          np.diag(np.ones(N-2), 2) * (-1) + \
          np.diag(np.ones(N-2), -2) * (-1)
    return mat / (12*h**2)
