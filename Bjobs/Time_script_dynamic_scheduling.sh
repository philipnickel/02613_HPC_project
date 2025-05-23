#!/bin/bash
#BSUB -J Python
#BSUB -q hpc
#BSUB -W 15
#BSUB -R "select[model == XeonGold6226R]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -R "span[hosts=1]"
#BSUB -n 16
#BSUB -o Experiments/Dynamic_parallelized/python_cores16_parallized_%J.out
#BSUB -e Experiments/Dynamic_parallelized/python_cores16_parallized_%J.err

# Initialize Python environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

#Run Python script
time python Python_scripts/dynamic_parallized_script.py 20