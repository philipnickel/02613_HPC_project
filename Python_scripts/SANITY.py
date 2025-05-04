import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import cupy as cp
from Funcs.jacobi import jacobi, jacobi_numba, jacobi_cp, jacobi_cuda
import os

# === MMS Problem Definition ===
def manufactured_solution(nx, ny, Lx=1.0, Ly=1.0):
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # harmonic function: u = sin(pi x) * sinh(pi y)
    u_exact = np.sin(np.pi * X) * np.sinh(np.pi * Y)
    f = np.zeros_like(u_exact)  # Laplace's equation ⇒ zero RHS

    u_exact_interior = u_exact[1:-1, 1:-1]
    f_interior = f[1:-1, 1:-1]
    interior_mask = np.ones_like(u_exact_interior, dtype=bool)

    return u_exact, f_interior, interior_mask, dx, dy


# === Compute Residual Norm ===
def compute_mms_residual(u, f, dx, dy, interior_mask):
    
    lap = (
        u[:-2, 1:-1] + u[2:, 1:-1] +
        u[1:-1, :-2] + u[1:-1, 2:] -
        4 * u[1:-1, 1:-1]
    ) / dx**2

    return np.linalg.norm((lap - f)[interior_mask])



def mms_iterative(jacobi_func, name, u0, f, dx, dy, interior_mask,
                         iter_chunk=500, max_total_iter=100000, atol=1e-9, **kwargs):
    u = np.copy(u0)
    residuals = []
    total_iter = 0

    while total_iter < max_total_iter:
        u_result = jacobi_func(u, interior_mask, iter_chunk, atol=0.0, **kwargs)
        u = u_result[0] if isinstance(u_result, tuple) else u_result

        u_cpu = cp.asnumpy(u) if (cp and isinstance(u, cp.ndarray)) else u
        res = compute_mms_residual(u_cpu, f, dx, dy, interior_mask)

        residuals.append(res)
        total_iter += iter_chunk
        print(f"{name}: iter {total_iter:5d}, residual = {res:.3e}")
        if res < atol:
            break

    return u, residuals



# === Main Execution ===
if __name__ == "__main__":
    max_total_iter = 100000
    atol = 1e-8
    nx, ny = 50, 50

    u_exact, f, interior_mask, dx, dy = manufactured_solution(nx, ny)
    u0 = np.zeros((nx - 2, ny - 2))
    u0_padded = np.pad(u0, pad_width=1)
    u0_padded[0, :]  = u_exact[0, :]
    u0_padded[-1, :] = u_exact[-1, :]
    u0_padded[:, 0]  = u_exact[:, 0]
    u0_padded[:, -1] = u_exact[:, -1]

    results = {}

    u_numpy, res_numpy = mms_iterative(jacobi, "Jacobi (NumPy)",
                                              u0_padded, f, dx, dy, interior_mask)
    results["Jacobi (NumPy)"] = res_numpy

    u_numba, res_numba = mms_iterative(jacobi_numba, "Jacobi (Numba)",
                                              u0_padded, f, dx, dy, interior_mask,
                                              parallel=False)
    results["Jacobi (Numba)"] = res_numba

    u_numba_par, res_numba_par = mms_iterative(jacobi_numba, "Jacobi (Numba Parallel)",
                                                      u0_padded, f, dx, dy, interior_mask,
                                                      parallel=True)
    results["Jacobi (Numba Parallel)"] = res_numba_par

    u_cuda, res_cuda = mms_iterative(jacobi_cuda, "Jacobi (CUDA)",
                                            u0_padded, f, dx, dy, interior_mask)
    results["Jacobi (CUDA)"] = res_cuda

    u_cp, res_cp = mms_iterative(jacobi_cp, "Jacobi (CuPy)",
                                        u0_padded, f, dx, dy, interior_mask)
    results["Jacobi (CuPy)"] = res_cp

    # === Plot Results ===
    os.makedirs("Sanity_Plots", exist_ok=True)

    for label, res_list in results.items():
        plt.figure()
        plt.semilogy(np.arange(1, len(res_list) + 1) * 500, res_list, label=label)
        plt.title(f"MMS Residual Convergence - {label}")
        plt.xlabel("Iteration")
        plt.ylabel("Residual (log scale)")
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        filename = f"Sanity_Plots/{label.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(filename)
        plt.close()

