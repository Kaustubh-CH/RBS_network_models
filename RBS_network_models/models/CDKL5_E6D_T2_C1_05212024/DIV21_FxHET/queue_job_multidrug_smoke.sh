#!/bin/bash
#SBATCH -A m2043
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH -t 00:30:00                    # debug queue max walltime
#SBATCH --nodes=2                      # 2 nodes × 256 tasks = 512 procs total; pop_size=2 → 1 node/trial
#SBATCH --ntasks-per-node=256
#SBATCH --cpus-per-task=1
#SBATCH --image=adammwea/netsims_docker:v1
#SBATCH --threads-per-core=2
#SBATCH --hint=socket
#SBATCH --mail-type=ALL
#SBATCH -J multidrug_smoke_v1
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

export SLURM_CPU_BIND="cores"

bash /pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/config_and_run_multidrug_smoke.sh
