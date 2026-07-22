#!/bin/bash
# run_rerun_svgs.sh - env setup + srun wrapper to regenerate SVGs from saved trial cfgs.
# (invocation wrapper only; calls rerun_trial_from_cfg.py unchanged)
set -e

module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
module load cray-mpich
module load conda
conda activate preshifter

export LD_LIBRARY_PATH=$MPICH_DIR/ofi/gnu/$(gcc -dumpversion)/lib:$MPICH_DIR/gtl/lib:$LD_LIBRARY_PATH
export MPI_LIB_NRN_PATH=$(find $MPICH_DIR -name libmpi.so | head -1)
export MPICH_GPU_SUPPORT_ENABLED=0

cd /pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/_scripts

BASE=/pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/CDKL5_BasetoNBQX_V2_test/batch_runs/batch_2026-04-02_spiking_only

for SPEC in "gen_7/trial_7_cfg.json" "gen_6/trial_6_cfg.json"; do
    CFG="$BASE/$SPEC"
    OUT="$(dirname "$CFG")/rerun_svgs_exp_style"
    echo "==================================================================="
    echo "Re-running: $CFG"
    echo "Output dir: $OUT"
    echo "==================================================================="
    srun -n 32 nrniv -mpi -python rerun_trial_from_cfg_experimental_style.py "$CFG" --output-dir "$OUT"
done
