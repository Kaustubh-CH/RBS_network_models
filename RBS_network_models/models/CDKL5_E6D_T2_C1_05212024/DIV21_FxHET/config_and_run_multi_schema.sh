#!/bin/bash -l
#SBATCH -N 1
#SBATCH -t 4:30:00
#SBATCH -q regular
#SBATCH -J Optuna_multi_schema
#SBATCH -L SCRATCH,cfs
#SBATCH -C cpu
#SBATCH --output logs/%A_%a  # job-array encodding
#SBATCH --image=balewski/ubu20-neuron8:v5
#SBATCH --array 1-1 #a
#SBATCH -A m2043

# updated # aw 2025-04-21 03:36:06
cd /pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models
module load conda
conda activate preshifter

echo $MPICH_DIR

# two main subdirs under $MPICH_DIR hold the .so files:
export LD_LIBRARY_PATH=$MPICH_DIR/ofi/gnu/$(gcc -dumpversion)/lib:$MPICH_DIR/gtl/lib:$LD_LIBRARY_PATH

# tell NEURON exactly which libmpi to dlopen:
export MPI_LIB_NRN_PATH=$(find $MPICH_DIR -name libmpi.so | head -1)

# Error out if SCHEMA_NAME is not set
if [ -z "$SCHEMA_NAME" ]; then
    echo "Error: SCHEMA_NAME environment variable not set."
    exit 1
fi

python /pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/run_batch_multi_schema.py --schema_name "$SCHEMA_NAME"
