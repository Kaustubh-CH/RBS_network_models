from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.src.batch import batchEvol_v2 as batchEvol
# from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.src.evol_params_from_seed_v2 import params
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.src.evol_params_from_seed_v3_large_v3 import params
# from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.src.evol_params_from_seed_v3_large import params
# from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.src.evol_params_from_seed_trial389 import params
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.src.conv_params import conv_params
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.src.conv_params import mega_params
# /RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/src
#from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.seeds import seeds
#from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.seeds_2 import seeds
# from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.seeds_5 import seeds
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.seeds_6 import seeds
# from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.seeds_ap5nbqx import seeds
#from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.fitness_schema.schema_1 import fit_schema
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.fitness_schema.schema_v3 import fit_schema
# Drug-response ratio fitting (fitnessFunc_v2_drug) uses its own schema.
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.fitness_schema.schema_v2_drug import fit_schema as fit_schema_drug
# Multi-drug schema (schema_v3 + drug_response component). Active when --drugs is passed.
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.fitness_schema.schema_v3_drug import fit_schema as fit_schema_v3_drug

# from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.fitness_schema.schema_3 import fit_schema
import netpyne
import os
import shutil

try:
    from mpi4py import MPI
    print("MPI4PY is installed, running in parallel mode")
except ImportError:
    print("WARNING: mpi4py not installed, running in single process mode")
    print("this is fine if debugging in login node, but not for batch jobs")
    pass

# main ========================================================================================
import sys
import importlib
import argparse

# CLI: optional --T_target override (any flag-style argv is consumed here; the
# remaining positionals keep the legacy semantics — first = run name, second =
# params-module override).
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--T_target', type=float, default=None,
                     help='Override target window length in seconds (default: 20.0). '
                          'Wins over network_results.attrs["T_target_s"] from the target h5.')
_parser.add_argument('--drugs', nargs='*', default=[],
                     help='Drug condition names from DRUG_REGISTRY (e.g. ap5_nbqx '
                          'bicuculline). Each candidate evaluates baseline + every '
                          'listed drug per generation; fitness includes the '
                          'drug_response component compared against /drug_effects/<drug>/ '
                          'in the reference h5. Empty = baseline-only (legacy).')
_known, _rest = _parser.parse_known_args()
_positional = [a for a in _rest if not a.startswith('-')]

# Validate drug names early — fail before any batch dir is created.
if _known.drugs:
    from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.src.drug_perturbations \
        import validate_drug
    for _d in _known.drugs:
        validate_drug(_d)
    print(f"--drugs active: {_known.drugs} (swapping fit_schema to schema_v3_drug)")
    _fit_schema_active = fit_schema_v3_drug
else:
    _fit_schema_active = fit_schema

name = _positional[0] if len(_positional) > 0 else "CDKL5_seed_v3_large_v3_02_w1_v2"

if len(_positional) > 1:
    params_module_name = _positional[1]
    module_path = f"RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.src.{params_module_name}"
    print(f"Overwriting params using module: {module_path}")
    params_mod = importlib.import_module(module_path)
    params = params_mod.params

T_target_override = _known.T_target  # None unless --T_target N was passed
if T_target_override is not None:
    print(f"--T_target override active: T_target_s_override = {T_target_override} s")

kwargs = {
    'parameter_space': params,
    'batchFolder': (
        #'/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/Organoid_RTT_R270X/DIV112_WT/batch_runs'
        f'/pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/{name}/batch_runs'
        # '/pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/CDKL5_Mar_25_seed_v3_large_02_w1_v3'
        # '/pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/KCNT_Test_Mar_03_seed_params_nostd_v4/batch_runs'
        ),
    'reference_data_paths': { # for fitting against
        #'/global/homes/a/adammwea/pscratch/zoutputs/CDKL5-E6D_T2_C1_05212024/CDKL5-E6D_T2_C1_05212024/240611/M08029/Network/000091/network_analysis/well005/metrics.npy'
        # '/pscratch/sd/k/ktub1999/networkSimulations/z_analyzed_data/network_analysis/well000/metrics.npy'
        # '/pscratch/sd/k/ktub1999/networkSimulatons_Sonnet/experimental_data_v2.h5'
        '/pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/processed_experimental_targets/CDKL5_002_well001.h5'
        # '/pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/processed_experimental_targets/CDKL5_011Imm_well001.h5'
        # '/pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/_scripts/experimental_features_20.h5'
        },
    'runCfg_script_path': (
        #'/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/Organoid_RTT_R270X/DIV112_WT/src/init.py'
        '/pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/src/init.py'
        ),
    "conv_params": conv_params,
    "mega_params": mega_params,
    # NO SEEDS: fresh unbiased study to converge to a different minimum.
    "seeds": None,
    "fit_schema": _fit_schema_active,  # schema_v3_drug if --drugs, else schema_v3
    "drugs": list(_known.drugs),       # consumed by src/batch.py + src/init.py

    # --- fitnessFunc_v2_drug inputs (simulated drug-response ratio fitting) ---
    # Baseline ("previous") simulated network — FIXED across the whole search.
    # Each candidate is the post-drug network; the fitness is the post/pre ratio
    # of features compared to the experimental drug ratios in the reference h5
    # (/drug_effects/<drug_name>/, stamped by _scripts/extract_drug_effects.py).
    "baseline_data_path": (
        '/pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/'
        'CDKL5_seed_large_limited_tau_2_w1_focus_InhFR/batch_runs/'
        'batch_2026-04-02_spiking_only/gen_39/trial_39_data.pkl'
        # '/pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/CDKL5_bicc_test_20/batch_runs/batch_2026-04-02_spiking_only/gen_28/trial_28_cfg.json'
        ),
    # Drug condition group key inside /drug_effects/<drug_name>/ of the reference h5
    # (reference_data_paths above -> CDKL5_002_well000.h5 has /drug_effects/bicuculline).
    # "drug_name": 'bicuculline',
    "drug_name":'ap5_nbqx',

    "plot_sim": True,
    "maxiter_wait":200, # number of iter to wait for job completion. ~200*time_sleep(5s)=1000s(~16.7min); covers slow/dense hyperactive networks (which can run >12min) so they aren't cut to 10000, slight trim off the old 220.
    "use_v2_burst_scoring": True, # whether to use the new burst scoring method that includes timing MSE, or the old method that only looks at burst counts.
    # Multi-drug smoke test config:
    # maxiters = total Optuna trials. pop_size = concurrent worker processes
    # (each trial = baseline + each --drug sim, run by one worker at a time).
    "maxiters": 1000,
    "pop_size": 2,
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
    'tag': 'probLC100_2000',
    'batchLabel': 'batch_2026-07-14_noseed',  # fresh baseline study, NO seeds (different minimum)
    'T_target_s_override': T_target_override,
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