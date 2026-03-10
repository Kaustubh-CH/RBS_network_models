# bin/bash

# updated # aw 2025-04-21 03:36:06


# uncomment as needed - just need to reinstall since i moved repos
# pip install -e /global/homes/a/adammwea/dev/RBS_network_models
# pip install -e /global/homes/a/adammwea/dev/netpyne
# pip install -e /global/homes/a/adammwea/dev/MEA_Analysis

# load the correct compiler environment
# module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
# module load cray-mpich
# echo $MPICH_DIR

# two main subdirs under $MPICH_DIR hold the .so files:


# tell NEURON exactly which libmpi to dlopen:


# validate setup
# nrniv -mpi -python - <<EOF
# from neuron import h
# print("MPI load OK, h =", h)
# EOF

#
cd networkSimulations/RBS_network_models/
module load conda
conda activate preshifter

export LD_LIBRARY_PATH=$MPICH_DIR/ofi/gnu/$(gcc -dumpversion)/lib:$MPICH_DIR/gtl/lib:$LD_LIBRARY_PATH
export MPI_LIB_NRN_PATH=$(find $MPICH_DIR -name libmpi.so | head -1)
python /pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/run_batch.py

#  salloc -A m2043_g -q interactive -C gpu -t 04:00:00 --nodes=1 --gpus=1 --image=nersc/pytorch:ngc-21.08-v2