#!/bin/bash
#BSUB -J time_jacobi
#BSUB -q hpc
#BSUB -R "select[model == XeonGold6226R]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -B 
#BSUB -R "span[hosts=1]"
#BSUB -n 20
#BSUB -o Logs/outputs/time_jacobi_%J.out
#BSUB -e Logs/errors/time_jacobi_%J.err

# Initialize Python environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

#Run Python script
time python Python_scripts/original_script.py 10
