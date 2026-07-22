#!/bin/bash
# Re-run gen_49 trial_49 (regular) from its cfg JSON for a 1 HOUR active window
# (3600 s + 5 s cool-down from cfg = 3605 s total) and save spike times +
# connectivity.
#
# This is the long-run counterpart of run_rerun_spikes_conn_gen49_25s.sh.
#
# --fixed-dt IS REQUIRED AT THIS DURATION. With the cfg's default CVode the run
# dies at t = 250476 ms with "error test failed repeatedly or with |h| = hmin"
# (h had collapsed to 2.3e-06), taking the whole srun step down with exit 255.
#
# Measured throughput on Perlmutter CPU (trial_49, 533 cells, 30 s sim window):
#   CVode,    128 ranks -> real-time ratio 0.41  (73 s)
#   CVode,    256 ranks -> real-time ratio 0.46  (65 s)
#   fixed dt, 256 ranks -> real-time ratio 1.73  (17 s)
# The model is communication-bound past ~128 ranks, so 256 buys only ~11% under
# CVode. Fixed dt=0.025 ms is 3.8x faster AND stable, putting a 3605 s window at
# ~35 min of wall clock instead of ~2.2 h. Spike statistics agree closely with
# CVode (32394 vs 32770 spikes; E 0.255 vs 0.241 Hz; I 8.125 vs 8.056 Hz).
#
#   salloc -A m2043 -q interactive -C cpu -t 04:00:00 --nodes=2 \
#       --ntasks-per-node=128 --cpus-per-task=2
#   bash run_rerun_spikes_conn_gen49_1hr.sh
#
# When driving a detached `salloc --no-shell` allocation from another shell,
# export SRUN_JOBID=<id> first so srun attaches to it.
set -euo pipefail

module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
module load cray-mpich
module load conda
conda activate preshifter

cd /pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/_scripts

export LD_LIBRARY_PATH=$MPICH_DIR/ofi/gnu/$(gcc -dumpversion)/lib:$MPICH_DIR/gtl/lib:$LD_LIBRARY_PATH
export MPI_LIB_NRN_PATH=$(find $MPICH_DIR -name libmpi.so | head -1)
export MPICH_GPU_SUPPORT_ENABLED=0

SRUN_JOBID_ARG="${SRUN_JOBID:+--jobid=$SRUN_JOBID}"

GEN49=/pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/CDKL5_seed_large_limited_tau_2_w1_focus_InhFR/batch_runs/batch_2026-04-02_spiking_only/gen_49
OUT=./op_gen49_trial49_1hr
mkdir -p "$OUT"

echo "=== Run: regular trial_49 (propVelocity=8.82), 3600 s active window ==="
srun $SRUN_JOBID_ARG -N 2 -n 256 --cpu-bind=cores nrniv -mpi -python rerun_trial_save_spikes_conn.py \
    "$GEN49/trial_49_cfg.json" \
    --duration-seconds 3600 \
    --fixed-dt \
    --output-dir "$OUT/"

# Plots (login node, no NEURON/MPI needed). --crop-start-seconds 5 drops the
# startup transient, matching the 25 s run's analysis window.
python plot_spikes_conn.py \
    --output-dir "$OUT" \
    --label trial_49_rerun_3600s \
    --crop-start-seconds 5
