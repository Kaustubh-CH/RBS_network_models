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
# from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.seeds_6 import seeds
# from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.seeds_7 import seeds
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.seeds_8 import seeds
# from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.seeds_ap5nbqx import seeds
#from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.fitness_schema.schema_1 import fit_schema
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.fitness_schema.schema_v3 import fit_schema
# Baseline schema v2 — same structure as v3, different weights: firing_rate_error_inh
# 0.30 (vs 0.70 in v3), firing_rate_error_exc 0.30 (vs 0.10), num_spikes_error 0.40
# (vs 0.20), unit_metrics group 0.40 (vs 0.50), network_bursts 0.20 (vs 0.10).
# Selected with --fit_schema v2 (baseline runs only).
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.fitness_schema.schema_v2 import fit_schema as fit_schema_v2_baseline
# Baseline schema v4. Selected with --fit_schema v4 (baseline runs only).
# Fitness values are NOT comparable across schemas — always use a fresh
# --batch_label when switching.
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.fitness_schema.schema_v4 import fit_schema as fit_schema_v4_baseline
# Baseline schema v4_5. Selected with --fit_schema v4_5 (baseline runs only).
# Identical to v4 except the two unit_metrics firing-rate terms are rebalanced to
# an even split: firing_rate_error_exc 0.10 -> 0.50, firing_rate_error_inh
# 0.70 -> 0.50. Intent is to stop the search from winning on inhibitory firing
# alone while the excitatory term sits saturated and contributes no gradient.
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.fitness_schema.schema_v4_5 import fit_schema as fit_schema_v4_5_baseline
# schema_v4_5 with firing_rate_error_exc/inh max_val capped 20000 -> 200, for use
# with RBS_FR_ERROR_STAT=median_std where that term is unbounded above.
# Selected with --fit_schema v4_5_medstd.
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.fitness_schema.schema_v4_5_medstd import fit_schema as fit_schema_v4_5_medstd_baseline
# Drug-response ratio fitting (fitnessFunc_v2_drug) uses its own schema.
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.fitness_schema.schema_v2_drug import fit_schema as fit_schema_drug
# Multi-drug schema (schema_v3 + drug_response component). Active when --drugs is passed.
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.fitness_schema.schema_v3_drug import fit_schema as fit_schema_v3_drug
# Theoretical-drug schema (40% drug / 60% baseline, adds duration + amplitude ratios).
# Active when --drug_target theory is passed alongside --drugs.
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.fitness_schema.schema_v3_drug_theory import fit_schema as fit_schema_v3_drug_theory
# v3 theory variant: drug_response drops the network_burst ratios and adds
# burstlet_rate + pre_burstlet rate/duration (requires the v3 theory target).
# Active when --drug_target theory_v3 is passed.
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.fitness_schema.schema_v3_drug_theory_v3 import fit_schema as fit_schema_v3_drug_theory_v3

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
_parser.add_argument('--drug_target', choices=['exp', 'theory', 'theory_v3'], default='exp',
                     help='Which drug target to fit against. "exp" (default) uses the '
                          'ratios extracted from paired pre/post recordings and '
                          'schema_v3_drug (30%% drug / 70%% baseline). "theory" uses the '
                          'pharmacology-derived target built by '
                          '_scripts/build_theoretical_drug_target.py and '
                          'schema_v3_drug_theory (40%% drug / 60%% baseline, with burst '
                          'duration and amplitude ratios). Only meaningful with --drugs.')
_parser.add_argument('--maxiters', type=int, default=None,
                     help='Override total number of trials (default: value in kwargs below).')
_parser.add_argument('--pop_size', type=int, default=None,
                     help='Override number of concurrent worker processes.')
_parser.add_argument('--batch_label', type=str, default=None,
                     help='Override batchLabel (the batch_<date>_<tag> subdirectory).')
_parser.add_argument('--IE_fr_check', action='store_true',
                     help='Treat a completely silent E or I population as a dealbreaker '
                          '(returns maxFitness). OFF by default, and it must stay opt-in: '
                          'drug conditions like ap5_nbqx legitimately drive excitatory '
                          'firing to zero, and a full GABA block can silence the '
                          'inhibitory side — culling those would discard the behaviour '
                          'being fitted. Turn it ON for baseline runs, where a dead '
                          'population is degenerate rather than meaningful.')
_parser.add_argument('--fit_schema', choices=['v2', 'v3', 'v4', 'v4_5', 'v4_5_medstd'], default='v3',
                     help='Baseline fitness schema (ignored when --drugs is passed, which '
                          'selects a drug schema instead). "v3" (default) weights '
                          'firing_rate_error_inh at 0.70; "v2" rebalances to 0.30 inh / '
                          '0.30 exc / 0.40 num_spikes and doubles network_bursts to 0.20. '
                          'Fitness values are NOT comparable across schemas — use a fresh '
                          '--batch_label when switching.')
_parser.add_argument('--maxiter_wait', type=int, default=None,
                     help='Override how many time_sleep(5s) polls the orchestrator waits for a '
                          'trial before abandoning it. Default 50 = 250s. Raise it when trials '
                          'finish but are cut off before fitness is computed (gen dirs with a '
                          'cfg.json but no fitness.json).')
_parser.add_argument('--seed_cfg', nargs='*', default=None,
                     help='Explicit seed candidate(s): path(s) to trial_*_cfg.json. Overrides '
                          'the seeds module imported above, so a new seed does not require '
                          'editing imports. Note src/batch.py truncates to pop_size seeds, '
                          'and pop_size is always 1 here, so only the first is used. '
                          'Mutually exclusive with --no_seeds.')
_parser.add_argument('--no_seeds', action='store_true',
                     help='Start the Optuna study from scratch with no seed individuals '
                          '(kwargs["seeds"] = None). Use when the params module ranges have '
                          'moved and the seed cfgs no longer sit inside them.')
_known, _rest = _parser.parse_known_args()
_positional = [a for a in _rest if not a.startswith('-')]

# Theoretical drug target — same baseline unit_* groups as the experimental target,
# but /drug_effects/<drug>/ holds theory-derived ratios instead of measured ones.
THEORY_TARGET_PATH = (
    '/pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/'
    'processed_experimental_targets/CDKL5_002_well001_theory_bicuculline.h5'
)
# v3 theory target — same baseline groups, but /drug_effects/ additionally holds
# burstlet_rate + pre_burstlet_* ratios so schema_v3_drug_theory_v3 can score them.
THEORY_V3_TARGET_PATH = (
    '/pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/'
    'processed_experimental_targets/CDKL5_002_well001_theory_bicuculline_v3.h5'
)

# Validate drug names early — fail before any batch dir is created.
if _known.drugs:
    from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.src.drug_perturbations \
        import validate_drug
    for _d in _known.drugs:
        validate_drug(_d)
    if _known.drug_target == 'theory_v3':
        print(f"--drugs active: {_known.drugs} with THEORETICAL target v3 "
              f"(fit_schema -> schema_v3_drug_theory_v3; drug_response = pop_FR, exc, inh, "
              f"burstlet dur/amp/rate, pre_burstlet dur/rate; NO network_burst ratios)")
        _fit_schema_active = fit_schema_v3_drug_theory_v3
    elif _known.drug_target == 'theory':
        print(f"--drugs active: {_known.drugs} with THEORETICAL target "
              f"(fit_schema -> schema_v3_drug_theory, 40% drug / 60% baseline)")
        _fit_schema_active = fit_schema_v3_drug_theory
    else:
        print(f"--drugs active: {_known.drugs} (swapping fit_schema to schema_v3_drug)")
        _fit_schema_active = fit_schema_v3_drug
else:
    if _known.drug_target in ('theory', 'theory_v3'):
        raise SystemExit(
            "--drug_target theory/theory_v3 requires --drugs (e.g. --drugs bicuculline); "
            "without a drug condition there is no drug_response term to score."
        )
    if _known.fit_schema == 'v2':
        print("--fit_schema v2: baseline schema_v2 "
              "(inh 0.30 / exc 0.30 / num_spikes 0.40, network_bursts 0.20)")
        _fit_schema_active = fit_schema_v2_baseline
    elif _known.fit_schema == 'v4':
        print("--fit_schema v4: baseline schema_v4")
        _fit_schema_active = fit_schema_v4_baseline
    elif _known.fit_schema == 'v4_5':
        print("--fit_schema v4_5: baseline schema_v4_5 "
              "(v4 with firing_rate_error exc/inh rebalanced 0.50 / 0.50)")
        _fit_schema_active = fit_schema_v4_5_baseline
    elif _known.fit_schema == 'v4_5_medstd':
        print("--fit_schema v4_5_medstd: schema_v4_5 with firing_rate_error "
              "exc/inh max_val capped 20000 -> 200. Intended for "
              "RBS_FR_ERROR_STAT=median_std, where the std term is unbounded "
              "above and would otherwise let unit_metrics swamp the burst terms.")
        _fit_schema_active = fit_schema_v4_5_medstd_baseline
        if os.environ.get('RBS_FR_ERROR_STAT', 'mean').strip().lower() != 'median_std':
            print("  WARNING: --fit_schema v4_5_medstd without "
                  "RBS_FR_ERROR_STAT=median_std. The cap will bind on nothing, "
                  "since the mean-based firing-rate error saturates at 100.")
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

# Experimental target — the well currently being fit.
#   other options kept for reference:
#     .../processed_experimental_targets/CDKL5_011Imm_well001.h5
#     .../_scripts/experimental_features_20.h5
EXP_TARGET_PATH = (
    '/pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/'
    'processed_experimental_targets/CDKL5_002_well001.h5'
)
if _known.drug_target == 'theory_v3':
    _reference_target_path = THEORY_V3_TARGET_PATH
elif _known.drug_target == 'theory':
    _reference_target_path = THEORY_TARGET_PATH
else:
    _reference_target_path = EXP_TARGET_PATH
if _known.drug_target in ('theory', 'theory_v3'):
    if not os.path.isfile(_reference_target_path):
        raise SystemExit(
            f"Theoretical target not found: {_reference_target_path}\n"
            "Build it first:\n"
            "  python RBS_network_models/_scripts/build_theoretical_drug_target.py \\\n"
            f"    --baseline_target {EXP_TARGET_PATH} \\\n"
            f"    --output {THEORY_TARGET_PATH} \\\n"
            "    --drug_name bicuculline"
        )
    print(f"reference target (theoretical): {_reference_target_path}")

kwargs = {
    'parameter_space': params,
    'batchFolder': (
        #'/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/Organoid_RTT_R270X/DIV112_WT/batch_runs'
        f'/pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/{name}/batch_runs'
        # '/pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/CDKL5_Mar_25_seed_v3_large_02_w1_v3'
        # '/pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/KCNT_Test_Mar_03_seed_params_nostd_v4/batch_runs'
        ),
    # for fitting against. --drug_target theory swaps in the theoretical-ratio copy,
    # whose baseline unit_* groups are identical so only the drug term changes.
    'reference_data_paths': {_reference_target_path},
    'runCfg_script_path': (
        #'/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/Organoid_RTT_R270X/DIV112_WT/src/init.py'
        '/pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/src/init.py'
        ),
    "conv_params": conv_params,
    "mega_params": mega_params,
    "seeds": seeds,
    # "seeds": None,
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
    "maxiter_wait":50, # number of iter to wait for job completion. ~200*time_sleep(5s)=1000s(~16.7min); covers slow/dense hyperactive networks (which can run >12min) so they aren't cut to 10000, slight trim off the old 220.
    "use_v2_burst_scoring": True, # whether to use the new burst scoring method that includes timing MSE, or the old method that only looks at burst counts.
    # Multi-drug smoke test config:
    # maxiters = total Optuna trials. pop_size = concurrent worker processes
    # (each trial = baseline + each --drug sim, run by one worker at a time).
    "maxiters": 1000,
    "pop_size": 1,
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
    # 2026-08-10: was 'probLC100_2000', which no longer described the search --
    # evol_params_large_limited_tau_cited caps probLengthConst at [85, 400].
    # The old tag would have stamped "100_2000" onto folders searching 85-400.
    'tag': 'probLC85_400',
    'batchLabel': 'batch_2026-07-14_baseline',  # fresh baseline study (no drugs, no resume)
    'T_target_s_override': T_target_override,
    }                       # also added deal breaker. If any one neuron has zeron synaptic connections, it is a deal breaker.

# CLI overrides for run scale, so a debug-sized run doesn't need a file edit.
if _known.maxiters is not None:
    kwargs['maxiters'] = _known.maxiters
    print(f"--maxiters override: {_known.maxiters}")
if _known.pop_size is not None:
    kwargs['pop_size'] = _known.pop_size
    print(f"--pop_size override: {_known.pop_size}")
if _known.no_seeds and _known.seed_cfg:
    raise SystemExit("--no_seeds and --seed_cfg are mutually exclusive; pick one.")
if _known.no_seeds:
    kwargs['seeds'] = None
    print("--no_seeds: starting from scratch, no seed individuals injected")
if _known.seed_cfg:
    for _s in _known.seed_cfg:
        _p = _s if _s.endswith('_cfg.json') else _s + '_cfg.json'
        if not os.path.isfile(_p):
            raise SystemExit(f"--seed_cfg path not found: {_p}")
    kwargs['seeds'] = list(_known.seed_cfg)
    print(f"--seed_cfg: seeding from {kwargs['seeds']}")
if _known.IE_fr_check:
    kwargs['IE_fr_check'] = True
    print("--IE_fr_check: silent E or I population is a DEALBREAKER (returns maxFitness)")
if _known.maxiter_wait is not None:
    kwargs['maxiter_wait'] = _known.maxiter_wait
    print(f"--maxiter_wait override: {_known.maxiter_wait} "
          f"(~{_known.maxiter_wait * 5}s per trial before abandonment)")
if _known.batch_label is not None:
    kwargs['batchLabel'] = _known.batch_label
elif _known.drug_target in ('theory', 'theory_v3'):
    # Never let a theoretical-target run land in the same batch dir as a
    # baseline or experimental-target run — the results are not comparable.
    kwargs['batchLabel'] = 'batch_2026-07-20_theory_bicuculline'
print(f"batchLabel: {kwargs['batchLabel']}")

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