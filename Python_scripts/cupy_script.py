from os.path import join
import os
import sys

import numpy as np
import cupy as cp

from Funcs.load_data import load_data, load_data_cp
from Funcs.jacobi import jacobi, jacobi_numba, jacobi_cp
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
    all_u0 = cp.empty((N, 514, 514))
    all_interior_mask = cp.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data_cp(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    Algebraic_TOL = 1e-2

    all_u = cp.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi_cp(u0, interior_mask, MAX_ITER, Algebraic_TOL)
        all_u[i] = u
    
    out_save_dir = "simulated_data"
    os.makedirs(out_save_dir, exist_ok=True)
    for bui in range(all_u.shape[0]):
        cp.save(f"simulated_data/{building_ids[bui]}", all_u[bui])

    """
    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
    """