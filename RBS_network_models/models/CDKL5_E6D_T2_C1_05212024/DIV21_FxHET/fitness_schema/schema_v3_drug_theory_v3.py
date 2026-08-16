"""
Fitness Schema v3 + THEORETICAL drug-response — V3 METRIC SET
=============================================================

Identical to ``schema_v3_drug_theory.py`` in every baseline component; ONLY the
``drug_response`` metric set differs, per request:

  DROP  network_burst_rate_ratio, network_burst_duration_ratio,
        network_burst_amp_ratio        (all network-burst drug ratios removed)
  KEEP  pop_FR_ratio, exc_firing_rate_ratio, inh_firing_rate_ratio,
        burstlet_duration_ratio, burstlet_amp_ratio
  ADD   burstlet_rate_ratio, pre_burstlet_duration_ratio,
        pre_burstlet_rate_ratio

The three ADDED metrics are NOT present in the original theoretical target, so
this schema requires the v3 target
``processed_experimental_targets/CDKL5_002_well001_theory_bicuculline_v3.h5``
(built after adding the matching burstlet_rate / pre_burstlet_* ratios to
``_scripts/build_theoretical_drug_target.py``). Fitting this schema against the
old target would silently drop those three metrics.

NOTE: only ``drug_response`` changes. The baseline burst components
(``burstlets`` / ``network_bursts`` / ``pre_burstlets``) are untouched — the
request was to remove network-burst ratios from the *drug response*, not from
baseline scoring.

max_dev = |theory_ratio - 1.0|, kept in sync with THEORETICAL_RATIOS in
build_theoretical_drug_target.py:
  pop_FR 2.0, exc 2.2, inh 1.5, burstlet_dur 2.0, burstlet_amp 2.0,
  burstlet_rate 1.8, pre_burstlet_dur 2.0, pre_burstlet_rate 1.8.

Selected by ``run_batch.py --drug_target theory_v3 --drugs bicuculline``.
"""

import copy

# Reuse the theory schema verbatim, then override only drug_response.
from .schema_v3_drug_theory import (
    fit_schema as _base_fit_schema,
    DEFAULT_CONFIG,
    EXPERIMENTAL_UNIT_ATTRS,
)

fit_schema = copy.deepcopy(_base_fit_schema)

fit_schema['drug_response'] = {
    'weight': 0.40,
    'description': ('Simulated post/pre ratios vs THEORETICAL drug ratios — v3 metric set: '
                   'no network-burst ratios; adds burstlet_rate + pre_burstlet rate/duration.'),
    'loss': 'log_soft',
    'metrics': {
        'pop_FR_ratio': {
            'weight': 0.15, 'max_dev': 1.0,   # theory ratio 2.0
            'description': 'Population firing-rate ratio (post/pre)',
        },
        'exc_firing_rate_ratio': {
            'weight': 0.15, 'max_dev': 1.2,   # theory ratio 2.2
            'description': 'Excitatory mean firing-rate ratio (post/pre)',
        },
        'inh_firing_rate_ratio': {
            'weight': 0.15, 'max_dev': 0.5,   # theory ratio 1.5
            'description': 'Inhibitory mean firing-rate ratio (post/pre)',
        },
        'burstlet_duration_ratio': {
            'weight': 0.15, 'max_dev': 1.0,   # theory ratio 2.0
            'description': 'Burstlet duration ratio (post/pre); detector static=40',
        },
        'burstlet_amp_ratio': {
            'weight': 0.10, 'max_dev': 1.0,   # theory ratio 2.0
            'description': 'Burstlet amplitude ratio (post/pre)',
        },
        'burstlet_rate_ratio': {
            'weight': 0.10, 'max_dev': 0.8,   # theory ratio 1.8
            'description': 'Burstlet rate ratio (post/pre); detector static=40',
        },
        'pre_burstlet_duration_ratio': {
            'weight': 0.10, 'max_dev': 1.0,   # theory ratio 2.0
            'description': 'Pre-burstlet (permissive threshold) duration ratio (post/pre)',
        },
        'pre_burstlet_rate_ratio': {
            'weight': 0.10, 'max_dev': 0.8,   # theory ratio 1.8
            'description': 'Pre-burstlet (permissive threshold) rate ratio (post/pre)',
        },
    },
}


def validate_weights() -> bool:
    """Check the drug_response metric weights sum to ~1.0 and the 40/60 split holds."""
    import numpy as np
    ok = True
    total = sum(c['weight'] for c in fit_schema.values())
    if not np.isclose(total, 1.0):
        print(f"Warning: component weights sum to {total:.4f}, not 1.0")
        ok = False
    mtotal = sum(m['weight'] for m in fit_schema['drug_response']['metrics'].values())
    if not np.isclose(mtotal, 1.0):
        print(f"Warning: drug_response metric weights sum to {mtotal:.4f}, not 1.0")
        ok = False
    return ok


if __name__ == "__main__":
    print("Schema v3 THEORETICAL drug target — V3 metric set")
    print(f"drug_response metrics: {list(fit_schema['drug_response']['metrics'].keys())}")
    print(f"weights valid: {validate_weights()}")
