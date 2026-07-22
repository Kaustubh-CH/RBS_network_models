#!/bin/bash
#SBATCH -A m2043
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -t 04:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=256
#SBATCH --cpus-per-task=1
#SBATCH --image=adammwea/netsims_docker:v1
#SBATCH --threads-per-core=2
#SBATCH --hint=socket
#SBATCH --mail-type=ALL
#SBATCH -J CDKL5_FixedPropVel

# Bind CPUs properly
export SLURM_CPU_BIND="cores"

# Baseline (non-drug) optimization launch for the fixed-propVelocity run.
bash /pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/config_and_run_fixedpropvel.sh

# submit with:
# sbatch /pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/queue_job_fixedpropvel.sh
