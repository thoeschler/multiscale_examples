from pde import a

import numpy as np
import scipy.integrate as sci
from scipy.sparse import block_array
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve


class Patch:
    def __init__(self, left, center, right):
        self.left = left
        self.center = center
        self.right = right
    def __repr__(self):
        return f"Patch({self.left:.4f}, {self.center:.4f}, {self.right:.4f})"

def create_patches(mesh):
    N = mesh.size - 1
    patches = []

    patches.append(Patch(mesh[0], 0.5 * (mesh[0] + mesh[1]), mesh[1]))
    for patch_num in range(1, N):
        patches.append(Patch(mesh[patch_num-1], mesh[patch_num], mesh[patch_num+1]))
    patches.append(Patch(mesh[N-1], 0.5 * (mesh[N-1] + mesh[N]), mesh[N]))

    return patches

def hat_left(left, right):
    return lambda x: (x - left) / (right - left)

def hat_left_prime(left, right):
    return lambda x: 1.0 / (right - left)

def hat_right(left, right):
    return lambda x: 1.0 - (x - left) / (right - left)

def hat_right_prime(left, right):
    return lambda x: - 1.0 / (right - left)

def xi(left, center, right, number):
    match number:
        case 0:
            return lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0
        case 1:
            return lambda x: (x - left) * (x - center)
        case 2:
            return lambda x: (x - center) * (x - right)
        case _:
            raise RuntimeError()

def xi_prime(left, center, right, number):
    match number:
        case 0:
            return lambda x: np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
        case 1:
            return lambda x: 2 * x - center - left
        case 2:
            return lambda x: 2 * x - center - right
        case _:
            raise RuntimeError()

def basis_func_left(left, center, right, number, rightmost=False):
    return lambda x: (hat_left(left, center)(x) * xi(left, center, right, number)(x)) * np.logical_not(rightmost) + \
        (hat_left(left, right)(x) * xi(left, center, right, number)(x)) * rightmost

def basis_func_right(left, center, right, number, leftmost=False):
    return lambda x: (hat_right(center, right)(x) * xi(left, center, right, number)(x)) * np.logical_not(leftmost) + \
        (hat_right(left, right)(x) * xi(left, center, right, number)(x)) * leftmost

def basis_func_left_prime(left, center, right, number, rightmost=False):
    return lambda x: (
            hat_left_prime(left, center)(x) * xi(left, center, right, number)(x) + \
            hat_left(left, center)(x) * xi_prime(left, center, right, number)(x)
        ) * np.logical_not(rightmost) + \
        (
            hat_left_prime(left, right)(x) * xi(left, center, right, number)(x) + \
            hat_left(left, right)(x) * xi_prime(left, center, right, number)(x)
        ) * rightmost

def basis_func_right_prime(left, center, right, number, leftmost=False):
    return lambda x: (
            hat_right_prime(center, right)(x) * xi(left, center, right, number)(x) + \
            hat_right(center, right)(x) * xi_prime(left, center, right, number)(x)
        ) * np.logical_not(leftmost) + \
        (
            hat_right_prime(left, right)(x) * xi(left, center, right, number)(x) + \
            hat_right(left, right)(x) * xi_prime(left, center, right, number)(x)
        ) * leftmost


def assemble_stiffness_matrix_blocks_patch(patches: list[Patch], patch_num: int, eps):
    N = len(patches) - 1
    assert(patch_num > 0 and patch_num < N)
    patch_self = patches[patch_num]

    A_list = []

    if patch_num == N - 1:
        patch_nums = (patch_num - 1, patch_num) if N > 2 else (patch_num,)
    elif patch_num == 1:
        patch_nums = (patch_num, patch_num + 1) if N > 2 else (patch_num,)
    else:
        patch_nums = (patch_num - 1, patch_num, patch_num + 1)

    # compute matrix blocks
    nrows = ncols = 3
    for patch_num_other in patch_nums:
        A_block = np.zeros((nrows, ncols))
        patch_other = patches[patch_num_other]
        # left part of the patch
        if patch_num_other <= patch_num:
            basis_func_other_prime = basis_func_right_prime if patch_num != patch_num_other else basis_func_left_prime
            for test in range(nrows):
                for trial in range(ncols):
                    A_block[test, trial] += sci.quad(
                        lambda x: basis_func_other_prime(
                            patch_other.left, patch_other.center, patch_other.right, trial
                            )(x) * \
                            basis_func_left_prime(
                                patch_self.left, patch_self.center, patch_self.right, test
                            )(x) * a(x / eps),
                        patch_self.left,
                        min(patch_other.right, patch_self.center)
                    )[0]
        # right part of the patch
        if patch_num_other >= patch_num:
            basis_func_other_prime = basis_func_left_prime if patch_num != patch_num_other else basis_func_right_prime
            for test in range(nrows):
                for trial in range(ncols):
                    A_block[test, trial] += sci.quad(
                        lambda x: basis_func_other_prime(
                            patch_other.left, patch_other.center, patch_other.right, trial
                            )(x) * \
                            basis_func_right_prime(
                                patch_self.left, patch_self.center, patch_self.right, test
                            )(x) * a(x / eps),
                        max(patch_other.left, patch_self.center), 
                        patch_self.right
                        )[0]

        A_list.append(A_block)
    return A_list

def assemble_stiffness_matrix(patches, eps):
    N = len(patches) - 1

    # create matrix blocks for each patch
    blocks = []
    for patch_num in range(1, N):
        blocks_patch = assemble_stiffness_matrix_blocks_patch(patches, patch_num, eps)
  
        pre = (patch_num - 2) * [None] if patch_num > 1 else []
        post = (N - 1 - len(blocks_patch) - len(pre)) * [None]
        blocks.append(
            pre + blocks_patch + post
        )

    # if there is only a single block, i.e. N = 2
    if len(blocks) == 1: return blocks[0][0]

    return block_array(blocks).tocsr()

def assemble_rhs(mesh, patches: list[Patch], bcs, eps):
    bc_left, bc_right = bcs
    N = mesh.size - 1
    rhs = np.zeros(3 * (N - 1))

    # bc left
    patch_left = patches[0]
    patch_right = patches[1]
    for test in range(3):
        index = test
        rhs[index] -= bc_left * sci.quad(
            lambda x: basis_func_right_prime(
                    patch_left.left, patch_left.center, patch_left.right,
                    0, leftmost=True
                )(x) * \
                basis_func_left_prime(
                    patch_right.left, patch_right.center, patch_right.right,
                    test, rightmost=False
                )(x) * a(x / eps),
                patch_left.left,
                patch_left.right
            )[0]

    # bc right
    patch_left = patches[N - 1]
    patch_right = patches[N]
    for test in range(3):
        index = 3 * (N - 2) + test
        rhs[index] -= bc_right * sci.quad(
            lambda x: basis_func_right_prime(
                patch_left.left, patch_left.center, patch_left.right,
                test, leftmost=False
                )(x) * \
                basis_func_left_prime(
                    patch_right.left, patch_right.center, patch_right.right,
                    0, rightmost=True
                )(x) * a(x / eps),
                patch_right.left,
                patch_right.right
            )[0]

    return rhs

def dofmap(patch_num: int, N: int, local_dof: int):
    return (patch_num == 0) * local_dof + \
        (patch_num > 0) * (3 * patch_num - 2 + local_dof) - \
            local_dof * (patch_num == N) # artificial term to stay in bounds

def eval(mesh, patches: list[Patch], u, x):
    N = u.size // 3 + 1
    x = np.atleast_1d(x)

    # get number of patch in which x lies
    patch_nums = np.searchsorted(mesh, x)
    patch_nums = patch_nums * (patch_nums > 0) * (patch_nums < N + 1) + (patch_nums == 0) + N * (patch_nums == N + 1)
    result = np.zeros_like(x)

    # left
    left = mesh[patch_nums - 1]
    center = mesh[patch_nums] * (patch_nums < N) + patches[N].center * (patch_nums == N)
    right = mesh[(patch_nums + 1) * (patch_nums < N) + N * (patch_nums == N)]
    ndofs = 3 * (patch_nums < N) + (patch_nums == N)
    for dof in range(3):
        mask_left = dof < ndofs
        result += u[dofmap(patch_nums, N, dof)] * basis_func_left(left, center, right, dof, patch_nums == N)(x) * mask_left

    # right
    left = mesh[(patch_nums - 2) * (patch_nums > 1)]
    center = mesh[(patch_nums - 1)] * (patch_nums > 1) + patches[0].center * (patch_nums == 1)
    right = mesh[patch_nums]
    ndofs = 3 * (patch_nums > 1) + (patch_nums == 1)
    for dof in range(3):
        mask_right = dof < ndofs
        result += u[dofmap(patch_nums - 1, N, dof)] * basis_func_right(left, center, right, dof, patch_nums == 1)(x) * mask_right
    return result


def main(mesh, patches, bcs, eps):
    A = assemble_stiffness_matrix(patches, eps)
    b = assemble_rhs(mesh, patches, bcs, eps)

    N = mesh.size - 1
    if N == 2:
        u = solve(A, b)
    else:
        u = spsolve(A, b)
    u_pad = np.hstack((bcs[0], u, bcs[1]))

    return u_pad
