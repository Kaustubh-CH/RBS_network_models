#!/bin/bash -l
# FRESH BASELINE (no drugs, NO seeds) evolutionary run for an INTERACTIVE salloc
# on 1 CPU node. Companion to config_and_run_baseline.sh (which resumes the seeded
# study). This one starts a brand-new Optuna study with seeds=None so it converges
# to a DIFFERENT minimum, unbiased by the seed individuals.
#
# Launch (interactive QOS max wall = 4:00:00):
#   salloc -A m2043 -q interactive -C cpu -t 4:00:00 \
#     --nodes=1 --ntasks-per-node=256 --cpus-per-task=1 \
#     --threads-per-core=2 --hint=socket \
#     --image=adammwea/netsims_docker:v1 \
#     bash config_and_run_baseline_fresh.sh
set -euo pipefail

module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
module load cray-mpich
module load conda
conda activate preshifter

cd /pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models

export LD_LIBRARY_PATH=$MPICH_DIR/ofi/gnu/$(gcc -dumpversion)/lib:$MPICH_DIR/gtl/lib:$LD_LIBRARY_PATH
export MPI_LIB_NRN_PATH=$(find $MPICH_DIR -name libmpi.so | head -1)
export MPICH_GPU_SUPPORT_ENABLED=0

name="CDKL5_baseline_noseed_2026-07-14"
params="evol_params_large_limited_tau"

echo "=== FRESH BASELINE evol run (NO drugs, NO seeds) | name=$name params=$params ==="
python /pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/run_batch_noseed.py \
    "$name" "$params"
