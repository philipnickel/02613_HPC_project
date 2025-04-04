#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from os.path import join

def main():
    # Directory that contains your NPY files and building_ids.txt
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'

    # Read all building IDs from file
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    # Determine how many building IDs to process (default 1 if no command-line argument)
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    else:
        N = 1

    # Prepare an output directory to store the images
    out_dir = "raw_plots"
    os.makedirs(out_dir, exist_ok=True)

    # For each building ID up to N
    for bid in building_ids[:N]:
        domain_path = join(LOAD_DIR, f"{bid}_domain.npy")
        mask_path = join(LOAD_DIR, f"{bid}_interior.npy")

        # Load raw arrays
        domain = np.load(domain_path)
        interior_mask = np.load(mask_path)

        # Plot the domain
        fig1 = plt.figure()
        plt.imshow(domain, cmap='magma', vmin=0, vmax=25)
        plt.title(f"Building {bid}: Domain")
        plt.colorbar()
        out_path_domain = join(out_dir, f"{bid}_domain.pdf")
        plt.savefig(out_path_domain, dpi=150)
        plt.close(fig1)  # Close to free memory

        # Plot the interior mask (boolean)
        fig2 = plt.figure()
        plt.imshow(interior_mask.astype(int), cmap='gray')
        plt.title(f"Building {bid}: Interior Mask")
        plt.colorbar()
        out_path_mask = join(out_dir, f"{bid}_interior.pdf")
        plt.savefig(out_path_mask, dpi=150)
        plt.close(fig2)  # Close to free memory

        print(f"Saved plots to {out_path_domain} and {out_path_mask}")

if __name__ == "__main__":
    main()
