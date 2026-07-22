'''
Evolutionary parameter space — bicuculline drug search (CDKL5_DIV21)

Seeded from:
  z_simulated_data/CDKL5_seed_large_limited_tau_2_w1_focus_InhFR/batch_runs/
  batch_2026-04-02_spiking_only/gen_39/trial_39_cfg.json

Design
------
Fit how the network reacts to bicuculline (a GABA_A antagonist) on top of a
FIXED baseline network. Every parameter is pinned to its seed (cfg) value with a
zero-width range [val, val] EXCEPT:

  * GABA inhibitory weights : weightIE_GABA, weightII_GABA
        -> allowed to range from 0 up to 10% of the seed value
           (bicuculline blocks GABA_A, so effective inhibition is reduced)
  * Na conductances         : gnabar_E, gnabar_I
        -> allowed to move within +/-30% of the seed value
  * K  conductances (gK)    : gkbar_E, gkbar_I
        -> allowed to move within +/-30% of the seed value
  * connection probabilities: probEE, probEI, probIE, probII
        -> free in [0, 1]

All other parameters (morphology, AMPA/NMDA weights, time constants, std-devs,
propVelocity, probLengthConst) have range width 0, so the optimizer holds them
at the seed value — they will not drift from baseline.
'''
from netpyne import specs

version = 2.0


def _band(value, frac):
    """Symmetric +/- ``frac`` band around ``value`` -> [value*(1-frac), value*(1+frac)].
    Ordered low-high so it also works for negative seeds."""
    a, b = value * (1.0 - frac), value * (1.0 + frac)
    return [min(a, b), max(a, b)]


def _fixed(value):
    return [value, value]


# --- Seed values, from trial_39_cfg.json -------------------------------------
_SEED = {
    'propVelocity': 9.334647775939214,

    # morphology std-devs (homogeneous populations)
    'E_diam_stdev': 0.0, 'E_L_stdev': 0.0, 'E_Ra_stdev': 0.0,
    'I_diam_stdev': 0.0, 'I_L_stdev': 0.0, 'I_Ra_stdev': 0.0,

    # morphology means
    'E_diam_mean': 18.461833673231585,
    'E_L_mean':    783.648139902429,
    'E_Ra_mean':   147.97810435251898,
    'I_diam_mean': 69.99799831528878,
    'I_L_mean':    984.1459798653865,
    'I_Ra_mean':   954.3723496121231,

    # connection length constant (held fixed; not a per-pair probability)
    'probLengthConst': 3060.3182297361773,

    # connection probabilities (free in [0,1])
    'probEE': 0.7586569967248948,
    'probEI': 0.5213019877755236,
    'probIE': 0.77718483454371,
    'probII': 0.6935650191630289,

    # excitatory synaptic weights (AMPA blocked by NBQX, NMDA by AP5) — fixed here
    'weightEI_AMPA': 1162.6220281305914,
    'weightEE_AMPA': 6530.205657636594,
    'weightEI_NMDA': 437.7034670881377,
    'weightEE_NMDA': 8842.178576407048,

    # GABA inhibitory weights (bicuculline target) — +/-10%
    'weightIE_GABA': 2095.1973986874477,
    'weightII_GABA': 6889.649507557601,

    # Na conductances (+/-30%) + std-devs (fixed)
    'gnabar_E': 273.35390183786114, 'gnabar_E_std': 0.0,
    'gnabar_I': 389.8071146010529,  'gnabar_I_std': 0.0,

    # K conductances / gK (+/-30%) + std-devs (fixed)
    'gkbar_E': 14.132926901357976, 'gkbar_E_std': 0.0,
    'gkbar_I': 13.899014079194476, 'gkbar_I_std': 0.0,

    # synaptic time constants
    'tau1_AMPA': 0.1976844634854387, 'tau2_AMPA': 2.0138536835933927,
    'tau1_NMDA': 0.15009492326279883, 'tau2_NMDA': 86.71259570924833,
    'tau1_GABA': 3.233172838521705,  'tau2_GABA': 19.848332725702694,
}

# Band fractions for the varying parameters.
_GABA_FRAC = 0.10   # GABA weights:  0 .. 10% of seed
_COND_FRAC = 0.30   # Na & K conductances: +/-30% of seed


params = specs.ODict()

# Propagation — fixed
params['propVelocity'] = _fixed(_SEED['propVelocity'])

# Morphology (means + std-devs) — all fixed
params.update({
    'E_diam_stdev': _fixed(_SEED['E_diam_stdev']),
    'E_L_stdev':    _fixed(_SEED['E_L_stdev']),
    'E_Ra_stdev':   _fixed(_SEED['E_Ra_stdev']),
    'I_diam_stdev': _fixed(_SEED['I_diam_stdev']),
    'I_L_stdev':    _fixed(_SEED['I_L_stdev']),
    'I_Ra_stdev':   _fixed(_SEED['I_Ra_stdev']),

    'E_diam_mean': _fixed(_SEED['E_diam_mean']),
    'E_L_mean':    _fixed(_SEED['E_L_mean']),
    'E_Ra_mean':   _fixed(_SEED['E_Ra_mean']),
    'I_diam_mean': _fixed(_SEED['I_diam_mean']),
    'I_L_mean':    _fixed(_SEED['I_L_mean']),
    'I_Ra_mean':   _fixed(_SEED['I_Ra_mean']),
})

# Connection length constant — fixed
params['probLengthConst'] = _fixed(_SEED['probLengthConst'])

# Connectivity ----------------------------------------------------------------
params.update({
    # *** VARYING: connection probabilities (free in [0,1]) ***
    'probIE': [0, 1],
    'probEE': [0, 1],
    'probII': [0, 1],
    'probEI': [0, 1],

    # AMPA / NMDA excitatory weights — fixed
    'weightEI_AMPA': _fixed(_SEED['weightEI_AMPA']),
    'weightEE_AMPA': _fixed(_SEED['weightEE_AMPA']),
    'weightEI_NMDA': _fixed(_SEED['weightEI_NMDA']),
    'weightEE_NMDA': _fixed(_SEED['weightEE_NMDA']),

    # *** VARYING: GABA inhibitory weights, 0 .. 10% of seed (bicuculline target) ***
    'weightIE_GABA': [0.0, _SEED['weightIE_GABA'] * _GABA_FRAC],
    'weightII_GABA': [0.0, _SEED['weightII_GABA'] * _GABA_FRAC],
})

# Conductances ----------------------------------------------------------------
params.update({
    # *** VARYING: Na conductances, +/-30% of seed ***
    'gnabar_E':     _band(_SEED['gnabar_E'], _COND_FRAC),
    'gnabar_E_std': _fixed(_SEED['gnabar_E_std']),

    # *** VARYING: gK of excitatory neurons, +/-30% of seed ***
    'gkbar_E':      _band(_SEED['gkbar_E'], _COND_FRAC),
    'gkbar_E_std':  _fixed(_SEED['gkbar_E_std']),

    # *** VARYING: Na conductances, +/-30% of seed ***
    'gnabar_I':     _band(_SEED['gnabar_I'], _COND_FRAC),
    'gnabar_I_std': _fixed(_SEED['gnabar_I_std']),

    # *** VARYING: gK of inhibitory neurons, +/-30% of seed ***
    'gkbar_I':      _band(_SEED['gkbar_I'], _COND_FRAC),
    'gkbar_I_std':  _fixed(_SEED['gkbar_I_std']),
})

# Synaptic time constants — all fixed
params.update({
    'tau1_AMPA': _fixed(_SEED['tau1_AMPA']),
    'tau2_AMPA': _fixed(_SEED['tau2_AMPA']),
    'tau1_NMDA': _fixed(_SEED['tau1_NMDA']),
    'tau2_NMDA': _fixed(_SEED['tau2_NMDA']),
    'tau1_GABA': _fixed(_SEED['tau1_GABA']),
    'tau2_GABA': _fixed(_SEED['tau2_GABA']),
})
