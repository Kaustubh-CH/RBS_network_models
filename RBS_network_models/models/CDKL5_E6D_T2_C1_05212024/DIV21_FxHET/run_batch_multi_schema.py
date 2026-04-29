from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.src.batch import batchEvol_v2 as batchEvol
# from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.src.evol_params_from_seed_v2 import params
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.src.evol_params_large_limited_tau import params
# from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.src.evol_params_from_seed_v3_large import params
# from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.src.evol_params_from_seed_trial389 import params
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.src.conv_params import conv_params
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.src.conv_params import mega_params
# /RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/src
#from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.seeds import seeds
#from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.seeds_2 import seeds
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.seeds_5 import seeds
#from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.fitness_schema.schema_1 import fit_schema
# from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.fitness_schema.schema_v2 import fit_schema
# from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.fitness_schema.schema_3 import fit_schema

import netpyne
import os
import shutil
import argparse
import importlib

parser = argparse.ArgumentParser(description="Run Batch Optimization")
parser.add_argument('--schema_name', type=str, required=True, help='Name of the schema (e.g. schema_v1, schema_v5)')
args = parser.parse_args()

# Dynamically import fit_schema based on schema_name
schema_module_path = "RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.fitness_schema.multi_schema.{0}".format(args.schema_name)
schema_module = importlib.import_module(schema_module_path)
fit_schema = schema_module.fit_schema

# RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/fitness_schema/multi_schema/schema_v5.py
# RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.fitness_schema.multi_schema.schema_v5
print(fit_schema)
try:
    from mpi4py import MPI
    print("MPI4PY is installed, running in parallel mode")
except ImportError:
    print("WARNING: mpi4py not installed, running in single process mode")
    print("this is fine if debugging in login node, but not for batch jobs")
    pass

# main ========================================================================================
kwargs = {
    'parameter_space': params,
    'batchFolder': (
        #'/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/Organoid_RTT_R270X/DIV112_WT/batch_runs'
        '/pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/CDKL5_large_limited_tau_02_w1_v1_multi_score/batch_runs_{0}'.format(args.schema_name)
        # '/pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/CDKL5_Mar_25_seed_v3_large_02_w1_v3'
        # '/pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/KCNT_Test_Mar_03_seed_params_nostd_v4/batch_runs'
        ),
    'reference_data_paths': { # for fitting against
        #'/global/homes/a/adammwea/pscratch/zoutputs/CDKL5-E6D_T2_C1_05212024/CDKL5-E6D_T2_C1_05212024/240611/M08029/Network/000091/network_analysis/well005/metrics.npy'
        # '/pscratch/sd/k/ktub1999/networkSimulations/z_analyzed_data/network_analysis/well000/metrics.npy'
        # '/pscratch/sd/k/ktub1999/networkSimulatons_Sonnet/experimental_data_v2.h5'
        '/pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/processed_experimental_targets/CDKL5_002_well001.h5'
        # '/pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/_scripts/experimental_features_20.h5'
        },
    'runCfg_script_path': (
        #'/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/Organoid_RTT_R270X/DIV112_WT/src/init.py'
        '/pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/src/init.py'
        ),
    "conv_params": conv_params,
    "mega_params": mega_params,
    # "seeds": seeds,
    # "seeds": None,
    "fit_schema": fit_schema,
    "plot_sim": True,
    "maxiter_wait":40, # number of iter to wait for job completion
    "use_v2_burst_scoring": True, # whether to use the new burst scoring method that includes timing MSE, or the old method that only looks at burst counts.
    
    # tags
    # older tags before implementing in run_batch.py - previously implemented in src/batch.py in hacky way.
        #tag = 'test'
        #tag = 'BRandFRs' # 2025-05-19 18:04:50 just setting this up because I will start running multiple batches per day and need to distinguish
        #tag = 'BRandFRratios' # 2025-05-19 21:05:05 just targeting frs didnt go too well...need to start by getting ratios right I think...
        #tag = 'normBRandFRR_optBLandAmps'
        #tag = 'normBRandFRR_optBLandAmps_2'
        # tag = 'normBRandFRR_optBLandAmps_db' #aw 2025-05-25 22:20:20
        #tag = 'dealbreakers'
    
    # newer tags after implementing in run_batch.py
    #'tag': 'reduced_dealbreakers', # aw 2025-05-27 13:23:25 - only dealbreaker: baseline and burst amplitudes fits must be <1000
    #'tag': 'reduced_dealbreakers_2', # aw 2025-05-27 13:23:25 - only dealbreaker: baseline and burst amplitudes fits must be <1000 and cannot only be e or I firing
    #'tag': 'reduced_dealbreakers_3', # aw 2025-05-27 13:23:25 - only dealbreaker: cannot only be e or I firing
    #'tag': 'reduced_dealbreakers_4', # aw 2025-05-28 10:27:26 - only dealbreaker: cannot only be e or I firing
    
    # 2025-05-28 17:37:06 - seems like trying to optimize bursting characteristics and spiking characteristics at the same time is not working well,
    # so I will try to optimize spiking characteristics first - since it's more fundamental to the network activity, and then optimize bursting characteristics later.
    # I think this will work since I can normalize the spiking activity to resist degress and then optimize bursting characteristics based on that.
    'tag': 'spiking_only', # HACK: hacked the fitness function for this to work right now. Will need to fix later.
    'batchLabel': 'batch_2026-04-02', # Set the batchLabel to resume an earlier optuna study run instead of starting from gen 0
    }                       # also added deal breaker. If any one neuron has zeron synaptic connections, it is a deal breaker.

os.makedirs(kwargs['batchFolder'], exist_ok=True)
shutil.copy2(__file__, os.path.join(kwargs['batchFolder'], f"run_batch_{kwargs.get('batchLabel', 'latest')}.py"))

batchEvol(**kwargs)

# run options =======================================================================
'''
shifter --image=adammwea/axonkilo_docker:v7 /bin/bash
'''


''' **************************************************************************
# run in login node for testing/debugging

# then run with python debugger or python as needed
python -m pdb /global/homes/a/adammwea/workspace/aw_scripts/Organoid_RTT_R270X_models/DIV112_WT/run_batch_login.py
python /global/homes/a/adammwea/workspace/aw_scripts/Organoid_RTT_R270X_models/DIV112_WT/run_batch_login.py

'''

''' **************************************************************************
# run everything in interactive node - run each script, one at a time

# step 1:
bash ~/workspace/aw_scripts/network_model_development/Organoid_RTT_R270X/DIV112_WT/test_batch_config_interact_allocate.sh

# step 2:
bash ~/workspace/aw_scripts/network_model_development/Organoid_RTT_R270X/DIV112_WT/test_batch_config_interact_run.sh

'''