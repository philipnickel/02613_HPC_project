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
conda activate HPC

#Run Python script
time python orginal_script.py