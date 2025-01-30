from pde import a

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from scipy.integrate import quad



def hat_left(left, right):
    return lambda x: (x - left) / (right - left)

def hat_left_prime(left, right):
    return lambda x: 1.0 / (right - left)

def hat_right(left, right):
    return lambda x: 1.0 - (x - left) / (right - left)

def hat_right_prime(left, right):
    return lambda x: - 1.0 / (right - left)

def rfb_trial_function_left(left, right, eps):
    def rfb_trial_left(x):
        C = 1.0 / (eps * (np.cos(right / eps) - np.cos(left / eps)) -  2 * (right - left))
        D = C * (2.0 * left - eps * np.cos(left / eps))
        return C * (eps * np.cos(x / eps) - 2 * x) + D
    return rfb_trial_left

def rfb_trial_function_right(left, right, eps):
    return rfb_trial_function_left(right, left, eps)

def rfb_trial_function_left_prime(left, right, eps):
    def rfb_trial_left_prime(x):
        C = 1.0 / (eps * (np.cos(right / eps) - np.cos(left / eps)) -  2 * (right - left))
        return C * (- np.sin(x / eps) - 2)
    return rfb_trial_left_prime

def rfb_trial_function_right_prime(left, right, eps):
    return rfb_trial_function_left_prime(right, left, eps)

def assemble_stiffness_matrix_element(left, right, eps):
    A_ref = np.empty((2, 2), dtype=float)
    a_eps = lambda x: a(x / eps)
    A_ref[0, 0] = quad(
        lambda x: hat_right_prime(left, right)(x) * rfb_trial_function_right_prime(left, right, eps=eps)(x) * \
            a_eps(x),
        left, right
    )[0]
    A_ref[0, 1] = quad(
        lambda x: a_eps(x) * hat_right_prime(left, right)(x) * rfb_trial_function_left_prime(left, right, eps=eps)(x),
        left, right
    )[0]
    A_ref[1, 0] = quad(
        lambda x: a_eps(x) * hat_left_prime(left, right)(x) * rfb_trial_function_right_prime(left, right, eps=eps)(x),
        left, right
    )[0]
    A_ref[1, 1] = quad(
        lambda x: a_eps(x) * hat_left_prime(left, right)(x) * rfb_trial_function_left_prime(left, right, eps=eps)(x),
        left, right
    )[0]
    return A_ref

def sparse_index(el, row_loc, col_loc):
    return (3 * el + 2 * row_loc + col_loc)

def assemble_stiffness_matrix(mesh, eps):
    N = mesh.size - 1
    nnz = 3 * (N + 1) - 2
    data = np.zeros(nnz)

    for el, (a, b) in enumerate(zip(mesh[:-1], mesh[1:])):
        A_el = assemble_stiffness_matrix_element(a, b, eps)
        nrow, ncol = A_el.shape
        indices = sparse_index(el, np.repeat(np.arange(nrow), ncol), np.tile(np.arange(ncol), nrow))
        data[indices] += A_el.flatten()

    row = np.hstack((0, 3 * np.arange(N) + 2, nnz))
    col = np.vstack((np.arange(-1, N), np.arange(N + 1), np.arange(1, N + 2))).T.flatten()[1:-1]

    return csr_matrix(((data, col, row)), shape=(N + 1, N + 1))

def apply_bcs(A: csr_matrix, rhs, bcs):
    bc_left, bc_right = bcs

    rhs -= bc_left * A[:,0].toarray().squeeze() + bc_right * A[:,-1].toarray().squeeze()

    A_new = A[1:-1, 1:-1]
    rhs_new = rhs[1:-1]

    return A_new, rhs_new

def eval(mesh, u: np.ndarray, x: np.ndarray, eps):
    element_num = np.searchsorted(mesh, x)

    left = mesh[element_num - 1]
    right = mesh[element_num]

    result = (element_num > 0) * u[element_num - 1] * rfb_trial_function_right(left, right, eps)(x) + \
        u[element_num] * rfb_trial_function_left(left, right, eps)(x)

    return result


def main(mesh, bcs, eps):
    A = assemble_stiffness_matrix(mesh, eps)
    rhs = np.zeros(mesh.size)
    A_bc, rhs_bc = apply_bcs(A, rhs, bcs)

    u = spsolve(A_bc, rhs_bc)
    u_pad = np.hstack((bcs[0], u, bcs[1]))

    return u_pad
