#!/bin/bash
#BSUB -J time_final_solution
#BSUB -q c02613
#BSUB -R "select[model == XeonGold6226R]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB]"
#BSUB -B
#BSUB -R "span[hosts=1]"
#BSUB -n 4
#BSUB -W 01:30
#BSUB -o Experiments/Exodia/python_cores16_cuda_parallized_%J.out
#BSUB -e Experiments/Exodia/python_cores16_cuda_parallized_%J.err

# Initialize Python environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

#Run Python script
time python Python_scripts/dynamic_parallized_script_cuda.py 9143