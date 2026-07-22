'''
evolutionary parameter space for CDKL5_DIV21 project

evol_params_from_seed_v2.py (2026-03-03):
  Ranges are centred ±30 % around the best individual found at generation 87
  of the KCNT_Test_Mar_03_seed_params_nostd_v2_large run
  (trial_87_cfg.json).

  Seed values (for reference):
    propVelocity    =  3.042
    E_diam_mean     = 47.196     I_diam_mean     = 15.306
    E_L_mean        = 2486.10    I_L_mean        = 244.677
    E_Ra_mean       = 286.688    I_Ra_mean       =  95.005
    probLengthConst = 6527.197
    probEE=0.352  probEI=0.785  probIE=0.134  probII=0.403
    weightEE=4173.10  weightEI=1289.22  weightIE=1982.45  weightII=769.14
    gnabar_E=77.910  gkbar_E=2.262  gnabar_I=10.988  gkbar_I=5.130
    tau1_exc=0.558  tau2_exc=19.681  tau1_inh=16.809  tau2_inh=0.593

  Strategy:
    - All *_stdev / *_std remain [0, 0] (no per-cell variability)
    - Every other parameter gets [seed * 0.70 , seed * 1.30] (±30 %)
    - Probabilities are additionally clamped to [0, 1]
'''
from netpyne import specs

version = 'from_seed_v2'

params = specs.ODict()

# ---------------------------------------------------------------------------
# Propagation
# seed: 3.042  → ±30 % → [2.13, 3.95]
# ---------------------------------------------------------------------------
params['propVelocity'] = [2.13, 3.95]

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
    # E_diam_mean  seed=47.196  → [33.04, 61.35]
    'E_diam_mean': [33.04, 61.35],

    # E_L_mean     seed=2486.10 → [1740.27, 3231.93]
    'E_L_mean':    [1740.27, 3231.93],

    # E_Ra_mean    seed=286.688 → [200.68, 372.69]
    'E_Ra_mean':   [200.68, 372.69],

    # I_diam_mean  seed=15.306  → [10.71, 19.90]
    'I_diam_mean': [10.71, 19.90],

    # I_L_mean     seed=244.677 → [171.27, 318.08]
    'I_L_mean':    [171.27, 318.08],

    # I_Ra_mean    seed=95.005  → [66.50, 123.51]
    'I_Ra_mean':   [66.50, 123.51],
})

# ---------------------------------------------------------------------------
# Connection Probability Length Constant
# seed=6527.197 → [4569.04, 8485.36]
# ---------------------------------------------------------------------------
params['probLengthConst'] = [4569.04, 8485.36]

# ---------------------------------------------------------------------------
# Connectivity — probabilities  (±30 %, clamped to [0, 1])
# ---------------------------------------------------------------------------
params.update({
    # probEE  seed=0.352  → [0.247, 0.458]
    'probEE': [0.247, 0.458],

    # probEI  seed=0.785  → [0.550, 1.0]  (upper clamped from 1.021)
    'probEI': [0.550, 1.0],

    # probIE  seed=0.134  → [0.094, 0.174]
    'probIE': [0.094, 0.174],

    # probII  seed=0.403  → [0.282, 0.523]
    'probII': [0.282, 0.523],
})

# ---------------------------------------------------------------------------
# Connectivity — weights  (±30 %)
# ---------------------------------------------------------------------------
params.update({
    # weightEE  seed=4173.099 → [2921.17, 5425.03]
    # 'weightEE': [2921.17, 5425.03],

    # # weightEI  seed=1289.218 → [902.45, 1675.98]
    # 'weightEI': [902.45, 1675.98],

    # # weightIE  seed=1982.447 → [1387.71, 2577.18]
    # 'weightIE': [1387.71, 2577.18],

    # # weightII  seed=769.139  → [538.40, 999.88]
    # 'weightII': [538.40, 999.88],


    'weightEE': [2921.17, 5425.03],

    # weightEI  seed=1289.218 → [902.45, 1675.98]
    'weightEI': [0, 1000],

    # weightIE  seed=1982.447 → [1387.71, 2577.18]
    'weightIE': [200.0, 1000.0],

    # weightII  seed=769.139  → [538.40, 999.88]
    'weightII': [538.40, 999.88],

})

# ---------------------------------------------------------------------------
# Conductances — Means  (±30 %, std fixed at 0)
# ---------------------------------------------------------------------------
params.update({
    # gnabar_E  seed=77.910  → [54.54, 101.28]
    'gnabar_E':     [20, 200],
    'gnabar_E_std': [0, 0],

    # gkbar_E   seed=2.262   → [1.58, 2.94]
    'gkbar_E':      [1.58, 2.94],
    'gkbar_E_std':  [0, 0],

    # gnabar_I  seed=10.988  → [7.69, 14.28]
    'gnabar_I':     [7.69, 14.28],
    'gnabar_I_std': [0, 0],

    # gkbar_I   seed=5.130   → [3.59, 6.67]
    'gkbar_I':      [3.59, 6.67],
    'gkbar_I_std':  [0, 0],
})

# ---------------------------------------------------------------------------
# Synaptic Time Constants  (±30 %)
# ---------------------------------------------------------------------------
params.update({
    # tau1_exc  seed=0.558   → [0.39, 0.73]
    'tau1_exc': [0.39, 0.73],

    # tau2_exc  seed=19.681  → [13.78, 25.59]
    'tau2_exc': [13.78, 25.59],

    # tau1_inh  seed=16.809  → [11.77, 21.85]
    'tau1_inh': [0.5, 3.0],

    # tau2_inh  seed=0.593   → [0.41, 0.77]
    'tau2_inh': [5.0, 25.0],
})
