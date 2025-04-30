from os.path import join
import os
import sys

import numpy as np

from Funcs.load_data import load_data
from Funcs.jacobi import jacobi, jacobi_numba, jacobi_cuda, jacobi_cp
from Funcs.summary_stats import summary_stats


if __name__ == '__main__':
    # Load data
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    building_ids = building_ids[:N]

    # Load floor plans
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # Run jacobi iterations for each floor plan
    MAX_ITER = 25_000
    Alg_TOL = 1e-20

    all_u = np.empty_like(all_u0)
    all_residuals_cuda = np.empty((N, MAX_ITER))
    all_residuals_cp = np.empty((N, MAX_ITER))
    all_residuals_numba = np.empty((N, MAX_ITER))
    all_residuals_numba_parallel = np.empty((N, MAX_ITER))
    all_residuals_ref = np.empty((N, MAX_ITER))
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u, residuals_cuda = jacobi_cuda(u0, interior_mask, MAX_ITER, Alg_TOL, interval=100, save_residuals=True)
        u, residuals_cp = jacobi_cp(u0, interior_mask, MAX_ITER, Alg_TOL, interval=100, save_residuals=True)
        u, residuals_numba = jacobi_numba(u0, interior_mask, MAX_ITER, Alg_TOL, parallel=True, print_residual=True, save_residuals=True)
        u, residuals_numba_parallel = jacobi_numba(u0, interior_mask, MAX_ITER, Alg_TOL, parallel=False, print_residual=True, save_residuals=True)
        u, residuals_ref = jacobi(u0, interior_mask, MAX_ITER, Alg_TOL, print_residual=True, save_residuals=True)
        all_u[i] = u
        all_residuals_cuda[i] = residuals_cuda
        all_residuals_cp[i] = residuals_cp
        all_residuals_numba[i] = residuals_numba
        all_residuals_numba_parallel[i] = residuals_numba_parallel
        all_residuals_ref[i] = residuals_ref

    # Print the residuals one by one
    print("CUDA:")
    for i in range(N):
        for j in range(len(all_residuals_cuda[i])):
            print(f"Building {i}, iteration {j}: {all_residuals_cuda[i][j]}")
    
    print("\nCP:")
    for i in range(N):
        for j in range(len(all_residuals_cp[i])):
            print(f"Building {i}, iteration {j}: {all_residuals_cp[i][j]}")
            
    print("\nNumba:")
    for i in range(N):
        for j in range(len(all_residuals_numba[i])):
            print(f"Building {i}, iteration {j}: {all_residuals_numba[i][j]}")
            
    print("\nNumba Parallel:")
    for i in range(N):
        for j in range(len(all_residuals_numba_parallel[i])):
            print(f"Building {i}, iteration {j}: {all_residuals_numba_parallel[i][j]}")
            
    print("\nRef:")
    for i in range(N):
        for j in range(len(all_residuals_ref[i])):
            print(f"Building {i}, iteration {j}: {all_residuals_ref[i][j]}")


    out_save_dir = "simulated_data"
    os.makedirs(out_save_dir, exist_ok=True)
    for bui in range(all_u.shape[0]):
        np.save(f"simulated_data/{building_ids[bui]}", all_u[bui])

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))

