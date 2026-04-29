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
params['propVelocity'] = [0.1, 10]

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
    'E_diam_mean': [1, 50],
    'E_L_mean':    [1, 1500],
    'E_Ra_mean':   [50, 200],

    'I_diam_mean': [4, 75],
    'I_L_mean':    [50, 1500],
    'I_Ra_mean':   [80, 1000],
})

# Connection Probability Length Constant
params['probLengthConst'] = [1, 7000]

# Connectivity Parameters
params.update({
    'probIE': [0, 1],
    'probEE': [0, 1],
    'probII': [0, 1],
    'probEI': [0, 1],

    # Excitatory synaptic weights split by receptor type
    # AMPA-mediated weights (blocked by NBQX)
    'weightEI_AMPA': [0, 2000],
    'weightEE_AMPA': [0, 20000],

    # NMDA-mediated weights (blocked by AP5)
    'weightEI_NMDA': [0, 2000],
    'weightEE_NMDA': [0, 20000],

    # GABA-mediated weights (blocked by Bicuculline / `Gabazine)
    'weightIE_GABA': [0, 3000],
    'weightII_GABA': [0, 20000],
})

# Sodium (gnabar) and Potassium (gkbar) Conductances
params.update({
    'gnabar_E':     [0, 500],
    'gnabar_E_std': [0, 0],   # Fixed to 0

    'gkbar_E':      [0, 20],
    'gkbar_E_std':  [0, 0],   # Fixed to 0

    'gnabar_I':     [0, 500],
    'gnabar_I_std': [0, 0],   # Fixed to 0

    'gkbar_I':      [0, 20],
    'gkbar_I_std':  [0, 0],   # Fixed to 0
})

# Synaptic Time Constants
params.update({
        # AMPA time constants (fast excitatory)
        'tau1_AMPA': [0.1, 100],
        'tau2_AMPA': [0.1, 500],

        # NMDA time constants (slow excitatory)
        'tau1_NMDA': [0.1, 100],
        'tau2_NMDA': [0.1, 500],

        # GABA time constants (inhibitory)
        'tau1_GABA': [0.1, 100],
        'tau2_GABA': [0.1, 100],
    })