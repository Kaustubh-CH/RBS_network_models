#!/bin/bash
#SBATCH --job-name=cnn_dataset_gen
#SBATCH --account=m2043
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=4
#SBATCH --image=balewski/ubu20-neuron8:v5
#SBATCH --output=dataset_gen_%j.out
#SBATCH --error=dataset_gen_%j.err

set -e

# Record start time
START_TIME=$(date +%s)

# ---------------------------------------------------------------------------
# Overridable environment variables
#   sbatch --export=N_SAMPLES=1000,SEED=42,BIN_MS=5,USE_LHS=1 slurm_submit.sh
# ---------------------------------------------------------------------------
N_SAMPLES=${N_SAMPLES:-10}
SEED=${SEED:-42}
BIN_MS=${BIN_MS:-5}
USE_LHS=${USE_LHS:-0}
DURATION_MS=${DURATION_MS:-25000}

N_SAMPLES=1
USE_LHS=0
BIN_MS=5
DURATION_MS=25000


# ---------------------------------------------------------------------------
# Paths — adjust these for your setup
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="/pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/src"
OUTPUT_DIR="/pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data"
OUTPUT_FILE="${OUTPUT_DIR}/cnn_dataset_${SLURM_JOB_ID}.h5"

# ---------------------------------------------------------------------------
# Validate MODEL_DIR exists before launching
# ---------------------------------------------------------------------------
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: MODEL_DIR does not exist: $MODEL_DIR" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Environment setup — MPI inside shifter
# ---------------------------------------------------------------------------
module load conda 2>/dev/null || true
conda activate preshifter 2>/dev/null || true
export LD_LIBRARY_PATH=$MPICH_DIR/ofi/gnu/$(gcc -dumpversion)/lib:$MPICH_DIR/gtl/lib:$LD_LIBRARY_PATH
export MPI_LIB_NRN_PATH=$(find $MPICH_DIR -name libmpi.so 2>/dev/null | head -1)

# ---------------------------------------------------------------------------
# Build Python command
# ---------------------------------------------------------------------------
LHS_FLAG=""
if [ "$USE_LHS" -eq 1 ]; then
    LHS_FLAG="--lhs"
fi

CMD="python -m sim_dataset_generator.run_dataset \
    --n_samples $N_SAMPLES \
    --output $OUTPUT_FILE \
    --bin_ms $BIN_MS \
    --model_dir $MODEL_DIR \
    --duration_ms $DURATION_MS \
    --overwrite \
    --no_mpi \
    $LHS_FLAG"

# ---------------------------------------------------------------------------
# Create output directory
# ---------------------------------------------------------------------------
mkdir -p $OUTPUT_DIR

# ---------------------------------------------------------------------------
# Print job summary
# ---------------------------------------------------------------------------
echo "=========================================="
echo "CNN Dataset Generator - SLURM Job $SLURM_JOB_ID"
echo "  N_SAMPLES:  $N_SAMPLES"
echo "  NODES:      $SLURM_NNODES"
echo "  TASKS:      $SLURM_NTASKS"
echo "  SEED:       $SEED"
echo "  BIN_MS:     $BIN_MS"
echo "  LHS:        $USE_LHS"
echo "  DURATION:   ${DURATION_MS} ms"
echo "  OUTPUT:     $OUTPUT_FILE"
echo "  MODEL_DIR:  $MODEL_DIR"
echo "  Started at: $(date)"
echo "=========================================="

# ---------------------------------------------------------------------------
# Run with MPI inside shifter container
# ---------------------------------------------------------------------------
echo "Running command: $CMD"
# srun --mpi=pmi2 $CMD

srun -n 1 $CMD
# ---------------------------------------------------------------------------
# Print completion summary
# ---------------------------------------------------------------------------
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
ELAPSED_MIN=$(( ELAPSED / 60 ))
ELAPSED_SEC=$(( ELAPSED % 60 ))

echo "=========================================="
echo "Job completed at $(date)"
echo "Elapsed time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "Output: $OUTPUT_FILE"
echo "=========================================="
