import numpy as np


def a(y):
    return 1.0 / (np.sin(y) + 2)

def u_ana(x, eps):
    return -eps * np.cos(x / eps) + 2 * x + 1
