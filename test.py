import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

import FD1d
import FD2d


def my_gaussian(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.exp(-alpha * x ** 2)


def d2_my_gaussian(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return (-2 * alpha + 4 * alpha ** 2 * x ** 2) * np.exp(-alpha * x ** 2)


def pot_harmonic(x: np.ndarray, omega: float = 1.0) -> np.ndarray:
    return 0.5 * omega ** 2 * x ** 2


def test1():
    """
    test the implementation of init_FD1d_grid()
    """
    x, h = FD1d.init_FD1d_grid(1.0, 2.0, 10)
    print(x)
    print(h)


def test2():
    """
    test the implementation of init_FD1d_grid()
    """
    A = -5.0
    B = 5.0
    Npoints = 21
    x, h = FD1d.init_FD1d_grid(A, B, Npoints)
    Npoints_plot = 200
    x_dense = np.linspace(A, B, Npoints_plot)
    plt.plot(x_dense, my_gaussian(x_dense), label='f(x)')
    plt.plot(x, my_gaussian(x), label='Sampled f(x)', marker='o')
    plt.legend()
    plt.show()


def test3():
    """
    test the implementation of build_D2_matrix_3pt()
    """
    a = FD1d.build_D2_matrix_3pt(8, 2.0)
    print(a)


def test4(N: int):
    """
    test the implementation of build_D2_matrix_3pt()
    using a Gaussian function
    """
    x_min = -5.0
    x_max = 5.0
    x, h = FD1d.init_FD1d_grid(x_min, x_max, N)
    fx = my_gaussian(x)
    Ndense = 200
    x_dense = np.linspace(x_min, x_max, Ndense)
    fx_dense = my_gaussian(x_dense)
    d2_fx_dense = d2_my_gaussian(x_dense)
    D2 = FD1d.build_D2_matrix_3pt(N, h)
    d2_fx = D2 @ fx
    plt.plot(x, fx, marker='o', label='sampled f(x)')
    plt.plot(x_dense, fx_dense, label='f(x)')
    plt.plot(x, d2_fx, marker='o', label='approx f''(x)')
    plt.plot(x_dense, d2_fx_dense, label='f''(x)')
    plt.legend()
    plt.grid()
    plt.show()


def test5(N: int):
    """
    test the Harmonic potential
    plot the eigenstates and eigenvalues
    """
    x_min = -5.0
    x_max = 5.0
    x, h = FD1d.init_FD1d_grid(x_min, x_max, N)
    D2 = FD1d.build_D2_matrix_3pt(N, h)
    Vpot = pot_harmonic(x, omega=1.0)
    Ham = -0.5 * D2 + np.diag(Vpot)
    evals, evecs = linalg.eig(Ham)
    Nstates = 5
    hbar = 1.0
    omega = 1.0
    print("Eigenvalues")
    for n in range(Nstates):
        E_ana = (2 * n + 1) * hbar * omega / 2
        print("{:5d} {:18.10f} {:18.10f} {:18.10e}".format(n, np.real(evals[n]), E_ana, abs(evals[n] - E_ana)))

    plt.plot(x, evecs[:, 0], label="1st eigenstate", marker="o")
    plt.plot(x, evecs[:, 1], label="2nd eigenstate", marker="o")
    plt.plot(x, evecs[:, 2], label="3rd eigenstate", marker="o")
    plt.show()


def test6():
    """
    test the implementation of build_D2_matrix_5pt()
    """
    a = FD1d.build_D2_matrix_5pt(8, 2.0)
    print(a)


def test7():
    """
    test the FD2dGrid constructor
    """
    a = FD2d.FD2dGrid((-5.0, 5.0), 3, (-5.0, 5.0), 4)
    print(a.x)
    print(a.y)
    print(a.r)


def test8():
    """
    test the mapping function
    """
    a = FD2d.FD2dGrid((-5.0, 5.0), 3, (-5.0, 5.0), 4)
    for i in range(a.Npoints):
        print(f"{i:3d} {a.r[0,i]:8.3f} {a.r[1,i]:8.3f}")


def main():
    # test4(51)
    # test5(51)
    # test7()
    test8()


if __name__ == "__main__":
    main()
