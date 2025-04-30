from os.path import join
import os
import sys

import numpy as np

from Funcs.load_data import load_data
from Funcs.jacobi import jacobi_cuda
from Funcs.summary_stats import summary_stats
from concurrent.futures import ProcessPoolExecutor, wait



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
    MAX_ITER = 20_000
    ALGB_TOL = 0.01
    INTERVAL = 500

    all_u = np.empty_like(all_u0)

    max_workers = int(os.environ.get("LSB_DJOB_NUMPROC", os.cpu_count()))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(jacobi_cuda, u0, mask, MAX_ITER, ALGB_TOL, INTERVAL)  
           for u0, mask in zip(all_u0, all_interior_mask)]

        wait(futures)

        all_u = np.array([future.result() for future in futures])
    
    out_save_dir = "parallized_simulated_data"
    os.makedirs(out_save_dir, exist_ok=True)
    for bui in range(all_u.shape[0]):
        np.save(f"parallized_simulated_data/{building_ids[bui]}", all_u[bui])

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))