#!/bin/bash
#BSUB -J sanity_check
#BSUB -q c02613
#BSUB -R "select[model == XeonGold6226R]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=2GB]"
#BSUB -B 
#BSUB -R "span[hosts=1]"
#BSUB -n 4
#BSUB -oo Logs/outputs/Sanity.out
#BSUB -eo Logs/errors/Sanity.err

# Initialize Python environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

#Run Python script
python Python_scripts/SANITY.py
