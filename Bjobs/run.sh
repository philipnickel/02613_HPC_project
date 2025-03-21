#!/bin/bash
#BSUB -J Python
#BSUB -q hpc
#BSUB -W 15
#BSUB -R "select[model == XeonGold6226R]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -R "span[hosts=1]"
#BSUB -n 10
#BSUB -o python_%J.out
#BSUB -e python_%J.err

# Initialize Python environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

#Run Python script
python Python_scripts/original_script.py