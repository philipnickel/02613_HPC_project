#!/bin/bash
#BSUB -J time_jacobi_cp
#BSUB -q c02613
#BSUB -R "select[model == XeonGold6226R]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=2GB]"
#BSUB -B 
#BSUB -R "span[hosts=1]"
#BSUB -n 4
#BSUB -o Logs/outputs/time_jacobi_cp%J.out
#BSUB -e Logs/errors/time_jacobi__cp%J.err

# Initialize Python environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

#Run Python script
time python Python_scripts/cupy_script.py 
