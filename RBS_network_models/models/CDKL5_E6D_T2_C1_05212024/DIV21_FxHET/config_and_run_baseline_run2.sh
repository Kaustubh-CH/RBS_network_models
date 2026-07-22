#!/bin/bash -l
# BASELINE (no drugs) evolutionary run for an INTERACTIVE salloc on 1 CPU node.
# SECOND, INDEPENDENT run: same proven CPU env / params / schema as
# config_and_run_baseline.sh, but with a DIFFERENT run name so it stores in a
# fresh folder (its own Optuna storage.db) instead of resuming the 07-14 study.
#   - NO --drugs flag  -> run_batch.py runs baseline (schema_v3 + fitnessFunc_v2)
#   - params module = evol_params_large_limited_tau (edited probLengthConst=[100,2000])
#   - maxiter_wait=50, maxiters=1000 (both read from run_batch.py)
#
# Output folder:
#   z_simulated_data/CDKL5_baseline_2026-07-16_run2/batch_runs/batch_2026-07-14_baseline_probLC100_2000/
#
# Launch (interactive QOS max wall = 4:00:00):
#   salloc -A m2043 -q interactive -C cpu -t 4:00:00 \
#     --nodes=1 --ntasks-per-node=256 --cpus-per-task=1 \
#     --threads-per-core=2 --hint=socket \
#     --image=adammwea/netsims_docker:v1 \
#     bash config_and_run_baseline_run2.sh
set -euo pipefail

module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
module load cray-mpich
module load conda
conda activate preshifter

cd /pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models

export LD_LIBRARY_PATH=$MPICH_DIR/ofi/gnu/$(gcc -dumpversion)/lib:$MPICH_DIR/gtl/lib:$LD_LIBRARY_PATH
export MPI_LIB_NRN_PATH=$(find $MPICH_DIR -name libmpi.so | head -1)
# CPU nodes have no GTL library; the orchestrator's own mpi4py MPI_Init aborts
# otherwise (the per-trial srun sets this inline too).
export MPICH_GPU_SUPPORT_ENABLED=0

name="CDKL5_baseline_2026-07-16_run2"
params="evol_params_large_limited_tau"

echo "=== BASELINE evol run #2 (NO drugs) | name=$name params=$params ==="
python /pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/run_batch.py \
    "$name" "$params"
