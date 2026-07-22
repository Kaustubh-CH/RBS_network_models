#!/bin/bash
# Re-run trial_135 from its cfg for 300 s and save spike times + connectivity.
# Run from an interactive allocation (salloc) or wrap in an sbatch job.
#
#   salloc -A m2043 -q interactive -C cpu -t 02:00:00 --nodes=1
#   bash run_rerun_spikes_conn_300s.sh
set -euo pipefail

module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
module load cray-mpich
module load conda
conda activate preshifter

cd /pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/_scripts

export LD_LIBRARY_PATH=$MPICH_DIR/ofi/gnu/$(gcc -dumpversion)/lib:$MPICH_DIR/gtl/lib:$LD_LIBRARY_PATH
export MPI_LIB_NRN_PATH=$(find $MPICH_DIR -name libmpi.so | head -1)
export MPICH_GPU_SUPPORT_ENABLED=0

CFG=/pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/CDKL5_seed_v4_02_w1_v1_multi_score/batch_runs_schema_v2/batch_2026-04-02_spiking_only/gen_135/trial_135_cfg.json

srun -n 128 nrniv -mpi -python rerun_trial_save_spikes_conn.py \
    "$CFG" \
    --duration-seconds 300 \
    --output-dir ./op_300s_trial135/
