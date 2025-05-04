import numpy as np
import time
import matplotlib.pyplot as plt


def create_test_case(nx, ny):
    u = np.zeros((nx + 2, ny + 2))  # ghost-padded
    u[:, -1] = 1.0  # Right BC = 1.0
    interior_mask = np.ones((nx, ny), dtype=bool)
    return u, interior_mask

def run_and_compare():
    nx, ny = 10, 10  # reduce if needed for GPU memory
    max_iter = 100
    atol = 1e-5

    # Create input grids
    u_ref, mask = create_test_case(nx, ny)
    u_numba, _ = create_test_case(nx, ny)
    u_numba_par_old, _ = create_test_case(nx, ny)
    u_numba_par_new, _ = create_test_case(nx, ny)
    u_cuda, _ = create_test_case(nx, ny)

    print("Running reference Python version...")
    t0 = time.perf_counter()
    u_ref = jacobi(u_ref, mask, max_iter, atol)
    t1 = time.perf_counter()
    time_ref = t1 - t0
    print(f"Time: {time_ref:.4f} s")

    print("Running Numba JIT version (serial)...")
    t0 = time.perf_counter()
    u_numba = jacobi_numba(u_numba, mask, max_iter, atol, parallel=False)
    t1 = time.perf_counter()
    time_numba = t1 - t0
    print(f"Time: {time_numba:.4f} s | Speedup: {(1 - time_numba / time_ref) * 100:.2f}%")

    print("Running Numba parallel version (original)...")
    t0 = time.perf_counter()
    u_numba_par_old = jacobi_numba(u_numba_par_old, mask, max_iter, atol, parallel=True, new=False)
    t1 = time.perf_counter()
    time_numba_par_old = t1 - t0
    print(f"Time: {time_numba_par_old:.4f} s | Speedup: {(1 - time_numba_par_old / time_ref) * 100:.2f}%")

    print("Running Numba parallel version (fused)...")
    t0 = time.perf_counter()
    u_numba_par_new = jacobi_numba(u_numba_par_new, mask, max_iter, atol, parallel=True, new=True)
    t1 = time.perf_counter()
    time_numba_par_new = t1 - t0
    print(f"Time: {time_numba_par_new:.4f} s | Speedup: {(1 - time_numba_par_new / time_ref) * 100:.2f}%")

    print("Running CUDA version...")
    t0 = time.perf_counter()
    u_cuda = jacobi_cuda(u_cuda,mask,  max_iter)
    t1 = time.perf_counter()
    time_cuda = t1 - t0
    print(f"Time: {time_cuda:.4f} s | Speedup: {(1 - time_cuda / time_ref) * 100:.2f}%")

    # Accuracy comparisons
    print("\nAccuracy comparisons vs. reference:")
    print("L2 (numba serial):       ", np.linalg.norm((u_ref - u_numba)))
    print("L2 (numba parallel old): ", np.linalg.norm((u_ref - u_numba_par_old)))
    print("L2 (numba parallel new): ", np.linalg.norm((u_ref - u_numba_par_new)))
    print("L2 (CUDA):               ", np.linalg.norm((u_ref - u_cuda)))

    # Visualize
    fig, axs = plt.subplots(1, 6, figsize=(20, 3))
    axs[0].imshow(u_ref, origin='lower'); axs[0].set_title("Reference")
    axs[1].imshow(u_numba, origin='lower'); axs[1].set_title("Numba Serial")
    axs[2].imshow(u_numba_par_old, origin='lower'); axs[2].set_title("Numba Parallel")
    axs[3].imshow(u_numba_par_new, origin='lower'); axs[3].set_title("Fused Parallel")
    axs[4].imshow(u_cuda, origin='lower'); axs[4].set_title("CUDA")
    residual = u_ref - u_cuda
    im = axs[5].imshow(residual, origin='lower', cmap='RdBu')
    axs[5].set_title("CUDA Residual")
    plt.colorbar(im, ax=axs[5])
    for ax in axs: ax.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_and_compare()
