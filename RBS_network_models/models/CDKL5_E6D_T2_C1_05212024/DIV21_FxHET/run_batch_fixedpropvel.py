"""
run_batch_fixedpropvel.py

Baseline (non-drug) evolutionary optimization launcher for the fixed-propVelocity
experiment. Mirrors the configuration that produced
    z_simulated_data/CDKL5_seed_large_limited_tau_2_w1_focus_InhFR/
    batch_runs/batch_2026-04-02_spiking_only/
(schema_v3, CDKL5_002_well001.h5 target, tag 'spiking_only', no seeds, no drugs)
with the ONLY change being the parameter space: evol_params_large_limited_tau,
whose propVelocity range is now [200, 600] um/ms (0.2-0.6 m/s, physiological).

Usage:
    python run_batch_fixedpropvel.py <run_name>
"""
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.src.batch import batchEvol_v2 as batchEvol
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.src.evol_params_large_limited_tau import params
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.src.conv_params import conv_params
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.src.conv_params import mega_params
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.fitness_schema.schema_v3 import fit_schema

import netpyne
import os
import shutil

try:
    from mpi4py import MPI
    print("MPI4PY is installed, running in parallel mode")
except ImportError:
    print("WARNING: mpi4py not installed, running in single process mode")
    pass

import sys

name = sys.argv[1] if len(sys.argv) > 1 else "CDKL5_seed_large_limited_tau_2_w1_FixedPropVel"

kwargs = {
    'parameter_space': params,
    'batchFolder': (
        f'/pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/{name}/batch_runs'
        ),
    'reference_data_paths': {
        '/pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/processed_experimental_targets/CDKL5_002_well001.h5'
        },
    'runCfg_script_path': (
        '/pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/src/init.py'
        ),
    "conv_params": conv_params,
    "mega_params": mega_params,
    # no seeds (baseline, matches focus_InhFR run)
    # no drugs (baseline)
    "fit_schema": fit_schema,
    "plot_sim": True,
    "maxiter_wait": 40,
    "use_v2_burst_scoring": True,
    "maxiters": 100000,
    'tag': 'spiking_only',
    'batchLabel': 'batch_2026-07-01',  # fresh study (folder: batch_2026-07-01_spiking_only)
    }

os.makedirs(kwargs['batchFolder'], exist_ok=True)
shutil.copy2(__file__, os.path.join(kwargs['batchFolder'], f"run_batch_{kwargs.get('batchLabel', 'latest')}.py"))

batchEvol(**kwargs)
