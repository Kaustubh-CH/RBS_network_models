#!/bin/bash -l
#SBATCH -N 1
#SBATCH -t 4:30:00
#SBATCH -q regular
#SBATCH -J Evolutionary
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

# uncomment as needed - just need to reinstall since i moved repos
# pip install -e /global/homes/a/adammwea/dev/RBS_network_models
# pip install -e /global/homes/a/adammwea/dev/netpyne
# pip install -e /global/homes/a/adammwea/dev/MEA_Analysis

# load the correct compiler environment
# module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
# module load cray-mpich
echo $MPICH_DIR

# two main subdirs under $MPICH_DIR hold the .so files:
export LD_LIBRARY_PATH=$MPICH_DIR/ofi/gnu/$(gcc -dumpversion)/lib:$MPICH_DIR/gtl/lib:$LD_LIBRARY_PATH

# tell NEURON exactly which libmpi to dlopen:
export MPI_LIB_NRN_PATH=$(find $MPICH_DIR -name libmpi.so | head -1)

# validate setup
# nrniv -mpi -python - <<EOF
# from neuron import h
# print("MPI load OK, h =", h)
# EOF
name="CDKL5_seed_large_02_w1_focus_InhFR_v2"
#
python /pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/run_batch.py $name evol_params_large