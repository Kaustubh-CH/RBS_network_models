#!/bin/bash
# Env setup + baseline (non-drug) optimization launch for the fixed-propVelocity run.
# Mirrors config_and_run.sh but points at run_batch_fixedpropvel.py and a new run name.

module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
module load cray-mpich
echo $MPICH_DIR

cd networkSimulations/RBS_network_models/
module load conda
conda activate preshifter

name=CDKL5_seed_large_limited_tau_2_w1_FixedPropVel

export LD_LIBRARY_PATH=$MPICH_DIR/ofi/gnu/$(gcc -dumpversion)/lib:$MPICH_DIR/gtl/lib:$LD_LIBRARY_PATH
export MPI_LIB_NRN_PATH=$(find $MPICH_DIR -name libmpi.so | head -1)
export MPICH_GPU_SUPPORT_ENABLED=0

python /pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/run_batch_fixedpropvel.py $name
