
#!/bin/bash
#BSUB -J time_all_solvers_10
#BSUB -q c02613
#BSUB -R "select[model == XeonGold6226R]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=15GB]"
#BSUB -B
#BSUB -R "span[hosts=1]"
#BSUB -n 4
#BSUB -W 00:20
#BSUB -oo Logs/outputs/time_all_solvers_10.out
#BSUB -eo Logs/errors/time_all_solvers_10.err


# Initialize Python environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

#Run Python script

echo "Running Original Implementation"
time python Python_scripts/original_script.py 10


echo "Running Numba Implementation"
time python Python_scripts/numba_script.py 10

echo "Running Parallel Numba Implementation"
time python Python_scripts/numba_script_parallel.py 10

echo "Running CuPy Implementation"
time python Python_scripts/cupy_script.py 10

echo "Running CUDA Implementation" 
time python Python_scripts/CUDA_script.py 10

echo "Running Dynamic Parallelized Cuda Implementation"
time python Python_scripts/dynamic_parallized_script_cuda.py 10