#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from os.path import join

def main():
    # Directory that contains your NPY files and building_ids.txt
    LOAD_DIR = 'parallized_simulated_data/'

    # Read all building IDs from file
    building_ids = [os.path.splitext(filename)[0] for filename in os.listdir(LOAD_DIR) if os.path.isfile(os.path.join(LOAD_DIR, filename))]

    # Determine how many building IDs to process (default 1 if no command-line argument)
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    else:
        N = 1

    # Prepare an output directory to store the images
    out_dir = "simulated_plots"
    os.makedirs(out_dir, exist_ok=True)

    # For each building ID up to N
    for bid in building_ids[:N]:
        simulated_data_path = join(LOAD_DIR, f"{bid}.npy")

        # Load raw arrays
        data = np.load(simulated_data_path)

        # Plot the domain
        fig1 = plt.figure()
        plt.imshow(data, cmap='magma', vmin=0, vmax=25)
        plt.title(f"Building {bid}: Simulated")
        plt.colorbar()
        out_path_domain = join(out_dir, f"{bid}_simulated.pdf")
        plt.savefig(out_path_domain, dpi=150)
        plt.close(fig1)  # Close to free memory


        print(f"Saved plots to {out_path_domain}")

if __name__ == "__main__":
    main()
