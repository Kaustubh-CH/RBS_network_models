#!/bin/bash
# Re-run gen_49 trial_49 (regular) and its high-propVelocity variant from their
# cfg JSONs at native duration (20 s + cool-down) and save spike times +
# connectivity for each.
#
#   salloc -A m2043 -q interactive -C cpu -t 02:00:00 --nodes=1
#   bash run_rerun_spikes_conn_gen49.sh
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

echo "=== Run 1/2: regular trial_49 (propVelocity=8.82) ==="
srun -n 128 nrniv -mpi -python rerun_trial_save_spikes_conn.py \
    "$GEN49/trial_49_cfg.json" \
    --sim-label trial_49 \
    --output-dir ./op_gen49_trial49/

echo "=== Run 2/2: high-propVelocity variant (propVelocity=500.82) ==="
srun -n 128 nrniv -mpi -python rerun_trial_save_spikes_conn.py \
    "$GEN49/trial_49_more profVelo_cfg.json" \
    --sim-label trial_49_more_profVelo \
    --output-dir ./op_gen49_trial49_more_profVelo/
