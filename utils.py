import numpy as np
from scipy.integrate import quad


def uniform_mesh(N):
    return np.linspace(0, 1, N + 1, endpoint=True)

def compute_linf_error(u_ana: callable, u_num: callable, x):
    return np.max(np.abs(u_ana(x) - u_num(x)))

def compute_discrete_L2_error(mesh, u_ana: callable, u_num: callable):
    left = mesh[:-1]
    right = mesh[1:]

    L2_error2 = 0.
    for l, r in zip(left, right):
        L2_error2 += quad(
            lambda x: (u_num(x) - u_ana(x)) ** 2,
            l, r
        )[0]

    return np.sqrt(L2_error2)
