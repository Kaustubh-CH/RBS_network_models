#!/bin/bash -l
# BICUCULLINE run against the THEORETICAL drug target, for an INTERACTIVE salloc
# on 1 CPU node. Same proven CPU env as config_and_run_baseline.sh, but:
#   - --drugs bicuculline        -> each trial = baseline sim + bicuculline sim
#   - --drug_target theory       -> schema_v3_drug_theory (40% drug / 60% baseline)
#                                   + CDKL5_002_well001_theory_bicuculline.h5
#   - debug scale (maxiters/pop_size) so this fits inside one 4 h allocation
#
# Build the target first (login node, once):
#   python RBS_network_models/_scripts/build_theoretical_drug_target.py \
#     --baseline_target processed_experimental_targets/CDKL5_002_well001.h5 \
#     --output          processed_experimental_targets/CDKL5_002_well001_theory_bicuculline.h5 \
#     --drug_name       bicuculline
#
# Launch — TWO STEPS. Do NOT pass this script as an argument to salloc.
# See .claude/skills/run-batch/SKILL.md for why (NetPyNE's mpi_direct cleanup
# only kills nrniv processes on the node the orchestrator itself runs on; put it
# on a login node and hung trials are never reaped, deadlocking the allocation).
#
#   # 1. bare salloc, no trailing command -> lands you on the compute node
#   salloc -A m2043 -q interactive -C cpu -t 4:00:00 \
#     --nodes=1 --ntasks-per-node=256 --cpus-per-task=1 \
#     --threads-per-core=2 --hint=socket \
#     --image=adammwea/netsims_docker:v1
#
#   # 2. confirm, then launch from that compute-node shell
#   hostname                                    # must print nidXXXXXX
#   bash config_and_run_theory_bicuculline.sh
#
# pop_size=1 always: this is an Optuna study, and study.optimize() is called
# without n_jobs, so trials run strictly sequentially. pop_size is an evol-mode
# parameter that the optuna path never reads -- raising it does nothing.
# set -euo pipefail

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

name="CDKL5_theory_bicuc_2026-07-20"
params="evol_params_large_limited_tau"

echo "=== THEORY bicuculline evol run | name=$name params=$params ==="
python /pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/run_batch.py \
    "$name" "$params" \
    --T_target 20 \
    --maxiters 40 \
    --pop_size 1 \
    --batch_label batch_2026-07-20_theory_bicuculline_v2 \
    --drug_target theory \
    --drugs bicuculline
