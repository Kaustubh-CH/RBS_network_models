'''
evolutionary parameter space for CDKL5_DIV21 project

v2_nostd (2026-03-03):
  Identical to evol_params.py version 2.0, EXCEPT all *_stdev / *_std
  parameters are fixed to [0, 0] (homogeneous cell populations, no
  per-cell variability).  All other ranges are untouched.
'''
from netpyne import specs

version = '2_nostd'

# Evolutionary Parameters
params = specs.ODict()

# Propagation Parameters
params['propVelocity'] = [0.1, 0.3]

# Morphology Parameters (Excitatory and Inhibitory Cells)
params.update({

    # --- Standard Deviations: fixed to 0 ---
    'E_diam_stdev': [0, 0],
    'E_L_stdev':    [0, 0],
    'E_Ra_stdev':   [0, 0],

    'I_diam_stdev': [0, 0],
    'I_L_stdev':    [0, 0],
    'I_Ra_stdev':   [0, 0],

    # --- Means: unchanged from v2 ---
    'E_diam_mean': [5, 30],
    'E_L_mean':    [50, 1000],
    'E_Ra_mean':   [70, 200],

    'I_diam_mean': [4, 15],
    'I_L_mean':    [50, 500],
    'I_Ra_mean':   [80, 200],
})

# Connection Probability Length Constant
params['probLengthConst'] = [1, 5000]

# Connectivity Parameters
params.update({
    'probIE': [0, 1],
    'probEE': [0, 1],
    'probII': [0, 1],
    'probEI': [0, 1],

    'weightEI': [0, 1000],
    'weightIE': [0, 1000],
    'weightEE': [0, 1000],
    'weightII': [0, 1000],
})

# Sodium (gnabar) and Potassium (gkbar) Conductances
params.update({
    'gnabar_E':     [0, 12],
    'gnabar_E_std': [0, 0],   # Fixed to 0

    'gkbar_E':      [0, 4],
    'gkbar_E_std':  [0, 0],   # Fixed to 0

    'gnabar_I':     [0, 10],
    'gnabar_I_std': [0, 0],   # Fixed to 0

    'gkbar_I':      [0, 5],
    'gkbar_I_std':  [0, 0],   # Fixed to 0
})

# Synaptic Time Constants
params.update({
    'tau1_exc': [0.1, 100],
    'tau2_exc': [0.1, 500],
    'tau1_inh': [0.1, 100],
    'tau2_inh': [0.1, 1000],
})
