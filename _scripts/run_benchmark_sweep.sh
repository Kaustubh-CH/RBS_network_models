#!/bin/bash
# Rank-scaling + build-vs-simulate sweep for the multi-drug sim cost.
# Run from INSIDE an salloc allocation (see plan). Mirrors config_and_run.sh env.
#
#   salloc -A m2043 -q interactive -C cpu -t 02:00:00 --nodes=1 \
#     --ntasks-per-node=256 --cpus-per-task=1 --threads-per-core=2 --hint=socket
#   bash run_benchmark_sweep.sh
set -u

REPO=/pscratch/sd/k/ktub1999/networkSimulations
SCRIPTS=$REPO/RBS_network_models/_scripts
BENCH=$SCRIPTS/benchmark_drug_sim.py

# --- model inputs (match run_batch.py) --------------------------------------
export FEATURE_DATA_PATH=$REPO/RBS_network_models/processed_experimental_targets/CDKL5_002_well001.h5
export BENCH_CFG_JSON=$REPO/z_simulated_data/CDKL5_multidrug_smoke_v1/batch_runs/batch_2026-06-18_multidrug_smoke_multidrug_smoke_v1/gen_6/trial_6_cfg.json
export BENCH_DRUGS=bicuculline,ap5_nbqx
export T_TARGET_S=10
export BENCH_TMP=$(mktemp -d)

# --- NEURON+MPI env (mirrors config_and_run.sh) -----------------------------
module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
module load cray-mpich conda
conda activate preshifter
export LD_LIBRARY_PATH=$MPICH_DIR/ofi/gnu/$(gcc -dumpversion)/lib:$MPICH_DIR/gtl/lib:$LD_LIBRARY_PATH
export MPI_LIB_NRN_PATH=$(find $MPICH_DIR -name libmpi.so | head -1)
export MPICH_GPU_SUPPORT_ENABLED=0

OUT=$SCRIPTS/benchmark_results
mkdir -p "$OUT"
CSV=$OUT/sweep_$(date +%Y%m%d_%H%M%S).csv
LOG=${CSV%.csv}.log
echo "mode,ranks,cond,create_s,simulate_s,n_spikes" > "$CSV"
echo "writing -> $CSV  (full log: $LOG)"

RANKS_LIST="${RANKS_LIST:-16 32 64 128 256}"
# When driving a detached `salloc --no-shell` allocation from another shell,
# srun must be told which job to use: export SRUN_JOBID=<id> before running.
SRUN_JOBID_ARG="${SRUN_JOBID:+--jobid=$SRUN_JOBID}"

run_one() {
  local mode=$1 R=$2
  echo "=== mode $mode @ $R ranks ===" | tee -a "$LOG"
  BENCH_MODE=$mode srun $SRUN_JOBID_ARG -n "$R" --cpu-bind=cores nrniv -mpi -python "$BENCH" 2>&1 \
    | tee -a "$LOG" | grep '^BENCH,' | sed 's/^BENCH,//' >> "$CSV"
}

# Sweep both modes at every rank count: mode A gives the scaling curve +
# create/simulate split; running B alongside gives the build-once timing AND
# the per-condition spike-count correctness check (A vs B must match).
for R in $RANKS_LIST; do
  run_one A "$R"
  run_one B "$R"
done

echo "=== done. results CSV: $CSV ==="
column -t -s, "$CSV"
rm -rf "$BENCH_TMP"
