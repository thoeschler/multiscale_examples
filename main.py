import finite_elements as fem
import generalized_finite_elements as gfem
import residual_free_bubbles as rfb
from pde import u_ana

from utils import *
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def convergence_rate(e1, e2, N1, N2):
    return np.log(e2 / e1) / np.log(N1 / N2)

def main(eps, plot=False):
    analytical_solution = lambda x: u_ana(x, eps)
    bcs = (-eps + 1.0, -eps * np.cos(1.0 / eps) + 3.0)

    iterations = 6
    N_values = 2 ** np.arange(1, iterations + 1)
    eL2_fem = np.empty(iterations)
    elinf_fem = np.empty(iterations)
    eL2_gfem = np.empty(iterations)
    elinf_gfem = np.empty(iterations)
    eL2_rfb = np.empty(iterations)
    elinf_rfb = np.empty(iterations)
    for count, N in enumerate(N_values):
        mesh = uniform_mesh(N)
        """
        Finite Elements
        """
        u_fem = fem.main(mesh, bcs, eps)
        eL2_fem[count] = compute_discrete_L2_error(
            mesh, u_ana=analytical_solution, u_num=interp1d(mesh, u_fem, 'linear')
        )
        elinf_fem[count] = compute_linf_error(
            u_ana=analytical_solution, u_num=interp1d(mesh, u_fem, 'linear'),
            x=uniform_mesh(100 * N)
        )

        """
        Generalized finite elements
        """
        patches = gfem.create_patches(mesh)
        u_gfem = gfem.main(mesh, patches, bcs, eps)
        eL2_gfem[count] = compute_discrete_L2_error(
            mesh, u_ana=analytical_solution, u_num=lambda x: gfem.eval(mesh, patches, u_gfem, x)
        )
        elinf_gfem[count] = compute_linf_error(
            u_ana=analytical_solution, u_num=lambda x: gfem.eval(mesh, patches, u_gfem, x),
            x=uniform_mesh(100 * N)
        )

        """
        Residual free bubbles
        """
        u_rfb = rfb.main(mesh, bcs, eps)
        eL2_rfb[count] = compute_discrete_L2_error(
            mesh, u_ana=analytical_solution, u_num=lambda x: rfb.eval(mesh, u_rfb, x, eps)
        )
        elinf_rfb[count] = compute_linf_error(
            u_ana=analytical_solution, u_num=lambda x: rfb.eval(mesh, u_rfb, x, eps),
            x=uniform_mesh(100 * N)
        )

        """
        Visualize
        """
        if plot:
            fig, ax = plt.subplots()
            x = uniform_mesh(100 * N)
            ax.plot(x, analytical_solution(x), label="u_ana")
            ax.plot(x, gfem.eval(mesh, patches, u_gfem, x), label="u_gfem")
            ax.plot(x, rfb.eval(mesh, u_rfb, x, eps), label="u_rfb")
            ax.legend()
            plt.show()

    """
    Convergence plot
    """
    fig, ax = plt.subplots()
    ax.loglog(N_values, eL2_fem, label="fem")
    print(f"fem convergence rates:\t{convergence_rate(eL2_fem[:-1], eL2_fem[1:], N_values[:-1], N_values[1:])}")
    ax.loglog(N_values, eL2_gfem, label="gfem")
    print(f"gfem convergence rates:\t{convergence_rate(eL2_gfem[:-1], eL2_gfem[1:], N_values[:-1], N_values[1:])}")
    ax.loglog(N_values, eL2_rfb, label="rfb")
    print(f"rfb convergence rates:\t{convergence_rate(eL2_rfb[:-1], eL2_rfb[1:], N_values[:-1], N_values[1:])}")
    ax.legend()
    ax.set_xlabel("N")
    ax.set_ylabel("L2 error")
    plt.savefig("convergence.pdf", dpi=200)

if __name__ == "__main__":
    main(0.05)
