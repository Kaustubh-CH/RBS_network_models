#!/bin/bash
# Multi-drug smoke test: baseline + ap5_nbqx + bicuculline conditions per candidate.
# Reference h5 (well001) has /drug_effects/{ap5_nbqx, bicuculline} stamped via
# extract_drug_effects.py.

# Compiler / MPI setup (must match queue_job's allocated env)
module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
module load cray-mpich
echo "MPICH_DIR = $MPICH_DIR"

cd networkSimulations/RBS_network_models/
module load conda
conda activate preshifter

# Tag the output run dir; batchLabel inside run_batch.py distinguishes resume from cold start.
name=CDKL5_multidrug_smoke_v1
params=evol_params_large_limited_tau

# NEURON MPI bindings (Perlmutter-specific — without these NEURON's MPI init silently fails)
export LD_LIBRARY_PATH=$MPICH_DIR/ofi/gnu/$(gcc -dumpversion)/lib:$MPICH_DIR/gtl/lib:$LD_LIBRARY_PATH
export MPI_LIB_NRN_PATH=$(find $MPICH_DIR -name libmpi.so | head -1)

# --drugs causes batch.py to set os.environ['DRUGS'] and init.py to loop:
# baseline → ap5_nbqx (scale_AMPA=scale_NMDA=0) → bicuculline (scale_GABA=0)
python /pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/run_batch.py \
    $name $params --T_target 20 --drugs ap5_nbqx bicuculline
