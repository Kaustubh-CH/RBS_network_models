#!/bin/bash
# Re-run gen_49 trial_49 (regular) from its cfg JSON for a 25 s active window
# (+5 s cool-down from cfg = 30 s total) and save spike times + connectivity.
#
# This is the 25 s counterpart of run_rerun_spikes_conn_gen49.sh (which used the
# native 20 s window). The extra 5 s lets the startup transient be cropped from
# the analysis (see run_rerun_spikes_conn_gen49_25s plotting step: the plotter is
# called with --crop-start-seconds 5, giving a clean 5-25 s steady-state window).
#
#   salloc -A m2043 -q interactive -C cpu -t 01:00:00 --nodes=1
#   bash run_rerun_spikes_conn_gen49_25s.sh
set -euo pipefail

module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
module load cray-mpich
module load conda
conda activate preshifter

cd /pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/_scripts

export LD_LIBRARY_PATH=$MPICH_DIR/ofi/gnu/$(gcc -dumpversion)/lib:$MPICH_DIR/gtl/lib:$LD_LIBRARY_PATH
export MPI_LIB_NRN_PATH=$(find $MPICH_DIR -name libmpi.so | head -1)
export MPICH_GPU_SUPPORT_ENABLED=0

GEN49=/pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/CDKL5_seed_large_limited_tau_2_w1_focus_InhFR/batch_runs/batch_2026-04-02_spiking_only/gen_49

echo "=== Run: regular trial_49 (propVelocity=8.82), 25 s active window ==="
srun -n 128 nrniv -mpi -python rerun_trial_save_spikes_conn.py \
    "$GEN49/trial_49_cfg.json" \
    --sim-label trial_49 \
    --duration-seconds 25 \
    --output-dir ./op_gen49_trial49_25s/
