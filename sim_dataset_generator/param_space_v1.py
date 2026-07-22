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
    ("propVelocity",       [0.1,  0.3]),       # m/s

    # ── Morphology StdDevs — Excitatory ────────────────────────────────────
    ("E_diam_stdev",       [0.0,  0.0]),       # µm
    ("E_L_stdev",          [0.0,  0.0]),       # µm
    ("E_Ra_stdev",         [0.0,  0.0]),       # Ω·cm

    # ── Morphology StdDevs — Inhibitory ────────────────────────────────────
    ("I_diam_stdev",       [0.0,  0.0]),       # µm
    ("I_L_stdev",          [0.0,  0.0]),       # µm
    ("I_Ra_stdev",         [0.0,  0.0]),       # Ω·cm

    # ── Morphology Means — Excitatory ──────────────────────────────────────
    ("E_diam_mean",        [5.0,  30.0]),      # µm
    ("E_L_mean",           [50.0, 1000.0]),    # µm
    ("E_Ra_mean",          [70.0, 200.0]),     # Ω·cm

    # ── Morphology Means — Inhibitory ──────────────────────────────────────
    ("I_diam_mean",        [4.0,  15.0]),      # µm
    ("I_L_mean",           [50.0, 500.0]),     # µm
    ("I_Ra_mean",          [80.0, 200.0]),     # Ω·cm

    # ── Connectivity Topology ──────────────────────────────────────────────
    ("probLengthConst",    [1.0,  5000.0]),    # µm
    ("probIE",             [0.0,  1.0]),       # I→E probability
    ("probEE",             [0.0,  1.0]),       # E→E probability
    ("probII",             [0.0,  1.0]),       # I→I probability
    ("probEI",             [0.0,  1.0]),       # E→I probability

    # ── Connectivity Weights ───────────────────────────────────────────────
    ("weightIE",           [0.0,  1000.0]),    # pA
    ("weightEE",           [0.0,  1000.0]),    # pA
    ("weightII",           [0.0,  1000.0]),    # pA
    ("weightEI",           [0.0,  1000.0]),    # pA

    # ── Ion Channel Conductances — Excitatory ──────────────────────────────
    ("gnabar_E",           [0.0,  12.0]),      # S/cm²
    ("gnabar_E_std",       [0.0,  0.0]),       # S/cm²
    ("gkbar_E",            [0.0,  4.0]),       # S/cm²
    ("gkbar_E_std",        [0.0,  0.0]),       # S/cm²

    # ── Ion Channel Conductances — Inhibitory ──────────────────────────────
    ("gnabar_I",           [0.0,  10.0]),      # S/cm²
    ("gnabar_I_std",       [0.0,  0.0]),       # S/cm²
    ("gkbar_I",            [0.0,  5.0]),       # S/cm²
    ("gkbar_I_std",        [0.0,  0.0]),       # S/cm²

    # ── Synaptic Time Constants ────────────────────────────────────────────
    ("tau1_exc",           [0.1,  100.0]),     # ms — excitatory rise
    ("tau2_exc",           [0.1,  500.0]),     # ms — excitatory decay
    ("tau1_inh",           [0.1,  100.0]),     # ms — inhibitory rise
    ("tau2_inh",           [0.1,  1000.0]),    # ms — inhibitory decay
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
