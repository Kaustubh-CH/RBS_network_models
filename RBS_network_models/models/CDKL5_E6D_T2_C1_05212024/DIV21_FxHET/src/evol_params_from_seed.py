'''
evolutionary parameter space for CDKL5_DIV21 project

evol_params_from_seed.py (2026-03-03):
  Ranges are centred ±30 % around the best individual found at generation 50
  of the KCNT_Test_Mar_03_starting_from_seed_smooth_v2_nostd run
  (trial_50_cfg.json).

  Seed values (for reference):
    propVelocity    =  4.347
    E_diam_mean     = 67.24     I_diam_mean     = 14.51
    E_L_mean        = 2764.69   I_L_mean        = 193.82
    E_Ra_mean       = 232.83    I_Ra_mean       =  98.99
    probLengthConst = 7631.29
    probEE=0.394  probEI=0.725  probIE=0.044  probII=0.542
    weightEE=5953.74  weightEI=2358.64  weightIE=2979.13  weightII=624.27
    gnabar_E=48.60  gkbar_E=1.502  gnabar_I=60.43  gkbar_I=6.363
    tau1_exc=731.52  tau2_exc=476.71  tau1_inh=234.50  tau2_inh=504.23

  Strategy:
    - All *_stdev / *_std remain [0, 0] (no per-cell variability)
    - Every other parameter gets [seed * 0.70 , seed * 1.30] (±30 %)
    - Probabilities are additionally clamped to [0, 1]
'''
from netpyne import specs

version = 'from_seed'

params = specs.ODict()

# ---------------------------------------------------------------------------
# Propagation
# seed: 4.347   → ±30 % → [3.04, 5.65]
# ---------------------------------------------------------------------------
params['propVelocity'] = [3.04, 5.65]

# ---------------------------------------------------------------------------
# Morphology — Standard Deviations (fixed at 0)
# ---------------------------------------------------------------------------
params.update({
    'E_diam_stdev': [0, 0],
    'E_L_stdev':    [0, 0],
    'E_Ra_stdev':   [0, 0],
    'I_diam_stdev': [0, 0],
    'I_L_stdev':    [0, 0],
    'I_Ra_stdev':   [0, 0],
})

# ---------------------------------------------------------------------------
# Morphology — Means  (±30 % around seed)
# ---------------------------------------------------------------------------
params.update({
    # E_diam_mean  seed=67.24   → [47.07, 87.41]
    'E_diam_mean': [47.07, 87.41],

    # E_L_mean     seed=2764.69 → [1935.28, 3594.10]
    'E_L_mean':    [1935.28, 3594.10],

    # E_Ra_mean    seed=232.83  → [162.98, 302.68]
    'E_Ra_mean':   [162.98, 302.68],

    # I_diam_mean  seed=14.51   → [10.16, 18.86]
    'I_diam_mean': [10.16, 18.86],

    # I_L_mean     seed=193.82  → [135.67, 252.07]
    'I_L_mean':    [135.67, 252.07],

    # I_Ra_mean    seed=98.99   → [69.29, 128.69]
    'I_Ra_mean':   [69.29, 128.69],
})

# ---------------------------------------------------------------------------
# Connection Probability Length Constant
# seed=7631.29 → [5341.90, 9920.68]
# ---------------------------------------------------------------------------
params['probLengthConst'] = [5341.90, 9920.68]

# ---------------------------------------------------------------------------
# Connectivity — probabilities  (±30 %, clamped to [0, 1])
# ---------------------------------------------------------------------------
params.update({
    # probEE  seed=0.394  → [0.276, 0.512]
    'probEE': [0.276, 1],

    # probEI  seed=0.725  → [0.508, 0.943]
    'probEI': [0.508, 1],

    # probIE  seed=0.044  → [0.031, 0.058]
    'probIE': [0.031, 1],

    # probII  seed=0.542  → [0.379, 0.705]
    'probII': [0.379, 1],
})

# ---------------------------------------------------------------------------
# Connectivity — weights  (±30 %)
# ---------------------------------------------------------------------------
params.update({
    # weightEE  seed=5953.74 → [4167.62, 7739.86]
    'weightEE': [4167.62, 7739.86],

    # weightEI  seed=2358.64 → [1651.05, 3066.23]
    'weightEI': [0, 3066.23],

    # weightIE  seed=2979.13 → [2085.39, 3872.87]
    # 'weightIE': [2085.39, 3872.87],
    'weightIE': [0, 1000],  

    # weightII  seed=624.27  → [436.99, 811.55]
    'weightII': [436.99, 811.55],
})

# ---------------------------------------------------------------------------
# Conductances — Means  (±30 %, std fixed at 0)
# ---------------------------------------------------------------------------
params.update({
    # gnabar_E  seed=48.60  → [34.02, 63.18]
    # 'gnabar_E':     [34.02, 63.18],
    'gnabar_E':     [0, 100],  
    'gnabar_E_std': [0, 0],

    # gkbar_E   seed=1.502  → [1.05, 1.95]
    # 'gkbar_E':      [1.05, 1.95],
    'gkbar_E':      [0,5],
    'gkbar_E_std':  [0, 0],

    # gnabar_I  seed=60.43  → [42.30, 78.56]
    # 'gnabar_I':     [42.30, 78.56],
    'gnabar_I':     [0, 100],
    'gnabar_I_std': [0, 0],

    # gkbar_I   seed=6.363  → [4.45, 8.27]
    # 'gkbar_I':      [4.45, 8.27],
    'gkbar_I':      [0, 10],
    'gkbar_I_std':  [0, 0],
})

# ---------------------------------------------------------------------------
# Synaptic Time Constants  (±30 %)
# ---------------------------------------------------------------------------
params.update({
    # tau1_exc  seed=731.52  → [512.06, 951.00]
    # 'tau1_exc': [512.06, 951.00],
    'tau1_exc':[0,10],

    # tau2_exc  seed=476.71  → [333.70, 619.72]
    # 'tau2_exc': [333.70, 619.72],
    'tau2_exc': [10, 20],

    # tau1_inh  seed=234.50  → [164.15, 304.85]
    # 'tau1_inh': [164.15, 304.85],
    'tau1_inh': [0, 20],

    # tau2_inh  seed=504.23  → [352.96, 655.50]
    'tau2_inh': [20, 255.50],
})
