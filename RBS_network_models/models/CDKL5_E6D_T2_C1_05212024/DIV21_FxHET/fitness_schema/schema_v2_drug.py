"""
Fitness Schema v2 — Drug-Response Ratio Aware
==============================================

Schema for ``fitnessFunc_v2_drug.py``.

Unlike ``schema_v2`` (which scores a single simulated network against a single
experimental recording), this schema scores how a simulated network *reacts to a
drug*. The fitness function runs two simulations:

    baseline (previous) sim   →  post-drug (current) sim

computes the post/pre ratio of a set of population-level features, and compares
those simulated ratios to the **experimental** drug-response ratios stamped into
the baseline target h5 under ``/drug_effects/<drug_name>/`` by
``_scripts/extract_drug_effects.py``.

The feature set here is exactly ``extract_drug_effects.RATIO_FEATURES`` — the
ratios that script writes and that ``plot_drug_ratio_histograms.py`` plots, so
the simulated and experimental sides are computed the same way (same E/I split,
same burst-detector params).

Each metric error is a squared log2-ratio distance::

    error = (log2(sim_ratio) - log2(exp_ratio)) ** 2

log space is used because ratios are multiplicative: a 2x increase and a 0.5x
decrease are equidistant from "no change" (1.0).

Author: Auto-generated
Date: May 2026
"""

import copy
import numpy as np
from typing import Dict


# =============================================================================
# BURST DETECTOR / CLASSIFICATION PARAMETERS
# (must match _scripts/extract_drug_effects.py so the experimental ratios in the
#  target h5 and the simulated ratios are computed under identical settings)
# =============================================================================

EPS = 1e-9

# Main burst detector kwargs (matches fitnessFunc_v2._run_burst_detector main pass
# and extract_drug_effects.MAIN_BURST_KW).
MAIN_BURST_KW = dict(base_threshold_static=40, min_burstlet_participation=0.05)

# Pre-burstlet detector kwargs (permissive threshold; matches
# extract_drug_effects.PRE_BURST_KW).
PRE_BURST_KW = dict(base_threshold_static=15, min_burstlet_participation=0.05)

# Top (1 - INHIB_QUANTILE) fraction of units by baseline firing rate -> inhibitory.
# Mirrors extract_drug_effects.classify_channels_top20 so the E/I-split firing
# rate ratios are comparable between sim and experiment.
INHIB_QUANTILE = 0.80

# The post/pre ratio features scored by this schema. Order/keys match
# extract_drug_effects.RATIO_FEATURES exactly.
RATIO_FEATURES = (
    "pop_FR_ratio",
    # "exc_firing_rate_ratio",
    # "inh_firing_rate_ratio",
    "burstlet_rate_ratio",
    "burstlet_duration_ratio",
    "network_burst_rate_ratio",
    "network_burst_duration_ratio",
    "superburst_rate_ratio",
    "superburst_duration_ratio",
    "pre_burstlet_rate_ratio",
    "pre_burstlet_duration_ratio",
    "mean_participation_ratio",
)


# =============================================================================
# FITNESS SCHEMA — one component holding per-ratio weights
# =============================================================================

fit_schema = {
    'drug_ratios': {
        'weight': 1.00,
        'description': 'Post/pre drug-response ratio comparison (sim vs experiment)',
        'metrics': {
            'pop_FR_ratio': {
                'weight': 0.15,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Population mean firing-rate ratio (post/pre)',
            },
            'exc_firing_rate_ratio': {
                'weight': 0.10,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Excitatory mean firing-rate ratio (post/pre)',
            },
            'inh_firing_rate_ratio': {
                'weight': 0.10,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Inhibitory mean firing-rate ratio (post/pre)',
            },
            'burstlet_rate_ratio': {
                'weight': 0.10,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Burstlet rate ratio (post/pre)',
            },
            'burstlet_duration_ratio': {
                'weight': 0.05,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Mean burstlet duration ratio (post/pre)',
            },
            'network_burst_rate_ratio': {
                'weight': 0.15,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Network-burst rate ratio (post/pre)',
            },
            'network_burst_duration_ratio': {
                'weight': 0.10,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Mean network-burst duration ratio (post/pre)',
            },
            'superburst_rate_ratio': {
                'weight': 0.05,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Superburst rate ratio (post/pre)',
            },
            'superburst_duration_ratio': {
                'weight': 0.05,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Mean superburst duration ratio (post/pre)',
            },
            'pre_burstlet_rate_ratio': {
                'weight': 0.05,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Pre-burstlet (permissive threshold) rate ratio (post/pre)',
            },
            'pre_burstlet_duration_ratio': {
                'weight': 0.05,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Pre-burstlet mean duration ratio (post/pre)',
            },
            'mean_participation_ratio': {
                'weight': 0.05,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Mean network-burst participation ratio (post/pre)',
            },
        },
    },
}


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    'main_burst_kwargs': dict(MAIN_BURST_KW),
    'pre_burstlet_kwargs': dict(PRE_BURST_KW),
    'inhib_quantile': INHIB_QUANTILE,
    'eps': EPS,
    # Scale applied to the weighted-mean squared-log2 error before clamping.
    'score_scale': 100.0,
    # Drug condition group key inside /drug_effects/<drug_name>/ in the target h5.
    # Override per-run via kwargs['drug_name']; None -> first group found.
    'drug_name': None,
    'recording_duration': 300.0,   # seconds (overridden from data)
    # Upper fitness clamp. Raised well above the ratio-score range so the
    # (unbounded) config-drift penalty is not capped/hidden by clamping.
    'max_fitness': 1_000_000.0,

    # --- Config-drift penalty -------------------------------------------------
    # A drug (e.g. bicuculline, a GABA_A antagonist) is modelled as changing ONLY
    # the GABA inhibitory weights. Every other optimized parameter of the
    # post-drug candidate should stay equal to the baseline ("previous") network.
    # The penalty adds  config_penalty_weight * Σ((cur-base)/base)^2 * config_penalty_scale
    # over all compared parameters EXCEPT those the drug is allowed to change.
    # NOTE: config_penalty_scale is its OWN scale (default 1.0) — the drift
    # penalty is deliberately NOT multiplied by the ratio score_scale (100).
    'config_penalty_weight': 1.0,
    'config_penalty_scale': 1.0,
    'exclude_params': None,   # None -> resolve from drug via DRUG_PARAM_EXCLUDE
}

# Default parameters the drug IS allowed to change (excluded from the drift
# penalty): GABA inhibitory->excitatory (IE) and inhibitory->inhibitory (II)
# weights — used when the drug isn't recognised in DRUG_PARAM_EXCLUDE.
PARAM_PENALTY_EXCLUDE = (
    'weightIE_GABA',
    'weightII_GABA',
)

# Per-drug mechanism -> which weights that drug is allowed to change (and are
# therefore NOT penalised for differing from baseline). Everything else is
# pinned to the baseline network by the drift penalty.
#   bicuculline (GABA_A antagonist)  -> may change GABA inhibitory weights
#   ap5_nbqx    (NMDA + AMPA block)  -> may change AMPA + NMDA excitatory weights
DRUG_PARAM_EXCLUDE = {
    'bicuculline': ('weightIE_GABA', 'weightII_GABA'),
    'ap5_nbqx': ('weightEE_AMPA', 'weightEI_AMPA', 'weightEE_NMDA', 'weightEI_NMDA'),
}


def _normalize_drug_key(name) -> str:
    """Lower-case, alphanumerics only (so 'AP5+NBQX' == 'ap5_nbqx' == 'ap5nbqx')."""
    return ''.join(ch for ch in str(name).lower() if ch.isalnum()) if name else ''


def exclude_params_for_drug(drug_name, default=PARAM_PENALTY_EXCLUDE):
    """Resolve which params a drug may change (excluded from the drift penalty).

    Direct (normalised) match against DRUG_PARAM_EXCLUDE first, then a substring
    fallback by mechanism class; otherwise returns ``default``.
    """
    if not drug_name:
        return tuple(default)
    key = _normalize_drug_key(drug_name)
    for k, v in DRUG_PARAM_EXCLUDE.items():
        if _normalize_drug_key(k) == key:
            return tuple(v)
    # mechanism-class substring fallback
    if any(t in key for t in ('ap5', 'apv', 'nbqx', 'cnqx', 'dnqx', 'kynurenic')):
        return DRUG_PARAM_EXCLUDE['ap5_nbqx']
    if any(t in key for t in ('bicuculline', 'bic', 'gabazine', 'sr95531', 'picrotoxin', 'ptx')):
        return DRUG_PARAM_EXCLUDE['bicuculline']
    return tuple(default)


# cfg keys that are never optimization parameters — ignored by the drift penalty
# when no explicit param_names list is supplied.
PARAM_PENALTY_DENYLIST = (
    'duration', 'duration_seconds', 'network_cool_down',
    'simLabel', 'seeds', 'filename', 'saveFolder', 'dt', 'recordStep',
)


# =============================================================================
# HELPER FUNCTIONS  (mirror schema_v2's API)
# =============================================================================

def get_component_weights() -> Dict[str, float]:
    """Return top-level component weights from the schema."""
    return {name: comp['weight'] for name, comp in fit_schema.items()}


def get_metric_weights(component: str = 'drug_ratios') -> Dict[str, float]:
    """Return per-ratio metric weights for a component."""
    if component not in fit_schema:
        raise ValueError(f"Unknown component: {component}")
    return {name: m['weight'] for name, m in fit_schema[component]['metrics'].items()}


def normalize_schema_weights() -> Dict:
    """Return a copy of fit_schema with all weight groups normalised to 1.0."""
    normalized = copy.deepcopy(fit_schema)

    total = sum(c['weight'] for c in normalized.values())
    if total > 0:
        for c in normalized.values():
            c['weight'] /= total

    for component in normalized.values():
        mtotal = sum(m['weight'] for m in component['metrics'].values())
        if mtotal > 0:
            for m in component['metrics'].values():
                m['weight'] /= mtotal

    return normalized


def validate_weights() -> bool:
    """Check that all weight groups sum to ~1.0."""
    total = sum(c['weight'] for c in fit_schema.values())
    ok = True
    if not np.isclose(total, 1.0):
        print(f"Warning: component weights sum to {total:.4f}, not 1.0")
        ok = False
    for cname, comp in fit_schema.items():
        mtotal = sum(m['weight'] for m in comp['metrics'].values())
        if not np.isclose(mtotal, 1.0):
            print(f"Warning: {cname} metric weights sum to {mtotal:.4f}, not 1.0")
            ok = False
    return ok


def get_full_config() -> Dict:
    """Return schema + config combined."""
    return {
        'schema': fit_schema,
        'config': DEFAULT_CONFIG,
        'ratio_features': RATIO_FEATURES,
    }


# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FITNESS SCHEMA v2 DRUG SUMMARY")
    print("=" * 60)
    for cname, comp in fit_schema.items():
        print(f"\n{cname.upper()} (weight={comp['weight']:.2f})")
        print(f"  {comp['description']}")
        for mname, m in comp['metrics'].items():
            print(f"    {mname:32s} w={m['weight']:.2f}  [{m['min_val']:.1f}, {m['max_val']:.1f}]")
    print(f"\nWeights valid: {validate_weights()}")
    print("✓ Drug schema module loaded.")
