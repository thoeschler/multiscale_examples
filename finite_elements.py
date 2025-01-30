from pde import a

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg


def quadrature_5p(f: callable, left, right):
    x = np.array([
        0.0000000000000000, -0.5384693101056831, 0.5384693101056831, -0.9061798459386640, 0.9061798459386640
    ])
    weights = np.array([
        0.5688888888888889, 0.4786286704993665, 0.4786286704993665, 0.2369268850561891, 0.2369268850561891
    ])

    return (right - left) / 2 * f(
        0.5 * (np.atleast_1d(right - left)[:,None] * x[None] + np.atleast_1d(left + right)[:,None])
        ).dot(weights)

def assemble_matrix_and_rhs(x: np.ndarray, eps: float, bcs: tuple):
    n = x.size
    integrate = quadrature_5p(lambda x: a(x / eps), x[:-1], x[1:])

    hl2 = (x[1:-1] - x[:-2]) ** 2
    hr2 = (x[2:] - x[1:-1]) ** 2
    assert(hl2.size == hr2.size == n - 2)

    data = np.vstack((
        np.hstack((0., - integrate[1:-1] / hl2[1:])),
        integrate[:-1] / hl2 + integrate[1:] / hr2,
        np.hstack((- integrate[1:-1] / hr2[:-1], 0.)),
    )).T.flatten()
    data = data[data != 0]
    assert(data.size == 3 * (n - 2) - 2)

    row_ind = np.repeat(np.arange(n-2), 3)[1:-1]
    col_ind = np.vstack((np.arange(-1, n-3), np.arange(0, n-2), np.arange(1, n-1))).T.flatten()[1:-1]

    A = csr_matrix((data, (row_ind, col_ind)), shape=(n-2, n-2))

    b = np.zeros(n-2)
    b[0] = integrate[0] * bcs[0] / hl2[0]
    b[-1] = integrate[-1] * bcs[1] / hr2[-1]

    return A, b

def main(mesh, bcs, eps):
    A, b = assemble_matrix_and_rhs(mesh, eps=eps, bcs=bcs)
    u, _ = cg(A, b, rtol=1e-16)
    
    u_pad = np.hstack((bcs[0], u, bcs[1]))

    return u_pad
