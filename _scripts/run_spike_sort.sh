#!/bin/bash
#SBATCH -A m2043_g
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -t 05:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --image=nersc/pytorch:ngc-21.08-v2


shifter 
source /home/miniconda3/etc/profile.d/conda.sh
conda activate preshifter
srun python3 /pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/_scripts/spikesort.py

