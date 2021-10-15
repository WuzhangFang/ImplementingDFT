import numpy as np

import FD1d

from scipy import sparse


class FD2dGrid:
    def __init__(self, x_domain, Nx, y_domain, Ny):
        self.Nx = Nx
        self.Ny = Ny
        self.x, self.hx = FD1d.init_FD1d_grid(x_domain[0], x_domain[1], Nx)
        self.y, self.hy = FD1d.init_FD1d_grid(y_domain[0], y_domain[1], Ny)
        self.dA = self.hx * self.hy
        self.Npoints = Nx * Ny
        self.r, self.idx_xy2ip, self.idx_ip2xy = self.__generate_idx()

    def __generate_idx(self):
        """
        idx_ip2xy and idx_xy2ip define the mapping between
        2D grids and linear grids.
        """
        r = np.zeros((2, self.Npoints))
        idx_ip2xy = np.zeros((2, self.Npoints))
        idx_xy2ip = np.zeros((self.Nx, self.Ny))
        ip = 0
        for j in range(self.Ny):
            for i in range(self.Nx):
                r[0, ip] = self.x[i]
                r[1, ip] = self.y[j]
                idx_ip2xy[0, ip] = i
                idx_ip2xy[1, ip] = j
                idx_xy2ip[i, j] = ip
                ip = ip + 1
        return r, idx_xy2ip, idx_ip2xy


def build_D2_matrix(grid: FD2dGrid):
    """
    take the FD2dGrid as input and return the D2 matrix
    """
    Nx = grid.Nx
    Ny = grid.Ny
    hx = grid.hx
    hy = grid.hy
    D2x = FD1d.build_D2_matrix_5pt(Nx, hx)
    D2y = FD1d.build_D2_matrix_5pt(Ny, hy)
    D2 = sparse.kron(D2x, sparse.eye(Ny)) + sparse.kron(sparse.eye(Nx), D2y)

    return D2
