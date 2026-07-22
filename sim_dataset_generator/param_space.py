"""
Parameter space definition for the NetPyNE simulation dataset generator.

Derived from evol_params.py v2.0.  This module is the **single source of truth**
for parameter names, ordering, and physical bounds used across the sampling,
simulation, and analysis pipeline.

Every other module that needs parameter metadata should import from here
rather than hard-coding its own copy.
"""

from collections import OrderedDict
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Canonical parameter bounds
# ---------------------------------------------------------------------------
# Keys   – parameter name (str)
# Values – [lower_bound, upper_bound] in physical units (see inline comments)

PARAM_BOUNDS: OrderedDict = OrderedDict([
    # ── Propagation ────────────────────────────────────────────────────────
    ("propVelocity",       [2.13, 3.95]),       # m/s

    # ── Morphology StdDevs — Excitatory ────────────────────────────────────
    ("E_diam_stdev",       [0.0,  0.0]),       # µm
    ("E_L_stdev",          [0.0,  0.0]),       # µm
    ("E_Ra_stdev",         [0.0,  0.0]),       # Ω·cm

    # ── Morphology StdDevs — Inhibitory ────────────────────────────────────
    ("I_diam_stdev",       [0.0,  0.0]),       # µm
    ("I_L_stdev",          [0.0,  0.0]),       # µm
    ("I_Ra_stdev",         [0.0,  0.0]),       # Ω·cm

    # ── Morphology Means — Excitatory ──────────────────────────────────────
    ("E_diam_mean",        [33.04, 61.35]),      # µm
    ("E_L_mean",           [1740.27, 3231.93]),    # µm
    ("E_Ra_mean",          [200.68, 372.69]),     # Ω·cm

    # ── Morphology Means — Inhibitory ──────────────────────────────────────
    ("I_diam_mean",        [10.71, 19.90]),      # µm
    ("I_L_mean",           [171.27, 318.08]),     # µm
    ("I_Ra_mean",          [66.50, 123.51]),     # Ω·cm

    # ── Connectivity Topology ──────────────────────────────────────────────
    ("probLengthConst",    [4569.04, 8485.36]),    # µm
    ("probIE",             [0.094, 0.174]),       # I→E probability
    ("probEE",             [0.247, 0.458]),       # E→E probability
    ("probII",            [0.282, 0.523]),       # I→I probability
    ("probEI",             [0.550, 1.0]),       # E→I probability

    # ── Connectivity Weights ───────────────────────────────────────────────
    ("weightIE",           [200.0, 1000.0]),    # pA
    ("weightEE",           [2921.17, 5425.03]),    # pA
    ("weightII",           [538.40, 999.88]),    # pA
    ("weightEI",           [0.0,  1000.0]),    # pA

    # ── Ion Channel Conductances — Excitatory ──────────────────────────────
    ("gnabar_E",           [20, 200]),      # S/cm²
    ("gnabar_E_std",       [0.0,  0.0]),       # S/cm²
    ("gkbar_E",            [1.58, 2.94]),       # S/cm²
    ("gkbar_E_std",        [0.0,  0.0]),       # S/cm²

    # ── Ion Channel Conductances — Inhibitory ──────────────────────────────
    ("gnabar_I",           [7.69, 14.28]),      # S/cm²
    ("gnabar_I_std",       [0.0,  0.0]),       # S/cm²
    ("gkbar_I",            [3.59, 6.67]),       # S/cm²
    ("gkbar_I_std",        [0.0,  0.0]),       # S/cm²

    # ── Synaptic Time Constants ────────────────────────────────────────────
    ("tau1_exc",           [0.39, 0.73]),     # ms — excitatory rise
    ("tau2_exc",           [13.78, 25.59]),     # ms — excitatory decay
    ("tau1_inh",           [0.5, 3.0]),     # ms — inhibitory rise
    ("tau2_inh",           [5.0, 25.0]),    # ms — inhibitory decay
])


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def get_param_names() -> List[str]:
    """Return the ordered list of parameter names."""
    return list(PARAM_BOUNDS.keys())


def get_bounds() -> np.ndarray:
    """Return bounds as a numpy array of shape ``(N_params, 2)``, dtype float64."""
    return np.array(list(PARAM_BOUNDS.values()), dtype=np.float64)


def get_n_params() -> int:
    """Return the total number of parameters."""
    return len(PARAM_BOUNDS)
