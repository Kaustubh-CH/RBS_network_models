"""
Fitness Schema v3 + THEORETICAL drug-response
=============================================

Variant of ``schema_v3_drug.py`` for fitting against a *theoretical* drug target
(one built by ``_scripts/build_theoretical_drug_target.py``) rather than ratios
extracted from a paired pre/post recording.

Three differences from ``schema_v3_drug.py``:

1. **40/60 split.** ``drug_response`` carries 0.40 of the fitness; every
   schema_v3 baseline component is scaled by 0.60 (was 0.30/0.70). The drug term
   is the thing being debugged, so it gets a larger share.

2. **Duration and amplitude are scored.** The experimental extractor has always
   written ``*_duration_ratio`` datasets into the target h5, but no schema ever
   listed them, so they were computed and thrown away. Burst amplitude
   (``burst_peak`` in the detector) was never extracted at all. Both are now
   first-class drug metrics -- amplitude and duration are the two features
   disinhibition moves most, and a drug schema built only from event *rates*
   cannot see either.

3. **``burst_amp_error`` in the baseline burst levels.** Scored by
   ``fitnessFunc_v2.compute_hierarchical_burst_scores_simple_v2`` as the relative
   error of mean per-unit burst peak. Schemas without this key resolve its weight
   to 0.0 and score exactly as before, so ``schema_v3.py`` and
   ``schema_v3_drug.py`` are unaffected.

``max_dev`` for each drug metric follows ``max_dev = |target_ratio - 1.0|``, so a
simulated ratio of 1.0 -- the drug did nothing -- incurs precisely full loss and
any movement toward the target reduces loss linearly. Keep this file's max_dev
values in sync with ``THEORETICAL_RATIOS`` in
``_scripts/build_theoretical_drug_target.py``.

Selected by ``run_batch.py --drug_target theory --drugs bicuculline``.
"""

import copy
import numpy as np
from typing import Any, Dict, Optional


# =============================================================================
# FITNESS SCHEMA — theoretical-drug variant of schema_v3
# =============================================================================

fit_schema = {
    # ------------------------------------------------------------------
    # 1. Per-unit spiking metrics (from xlsx / computed on-the-fly)
    # ------------------------------------------------------------------
    'unit_metrics': {
        'weight': 0.30,  # was 0.50 in schema_v3 (× 0.60)
        'description': 'Single-unit firing property comparison',
        'metrics': {
            'firing_rate_error': {
                'weight': 0.00,
                'min_val': 0.0,
                'max_val': 10000.0,
                'description': 'Legacy aggregate firing-rate error (kept for compatibility)',
            },
            'firing_rate_error_exc': {
                'weight': 0.10,
                'min_val': 0.0,
                'max_val': 20000.0,
                'description': 'Error in excitatory mean firing rate (Hz)',
            },
            'firing_rate_error_inh': {
                'weight': 0.70,
                'min_val': 0.0,
                'max_val': 20000.0,
                'description': 'Error in inhibitory mean firing rate (Hz)',
            },
            'num_spikes_error': {
                'weight': 0.20,
                'min_val': 0.0,
                'max_val': 130.0,
                'description': 'Normalized error in total spike counts across units',
            },
            'firing_range_error': {
                'weight': 0.00,
                'min_val': 0.0,
                'max_val': 20.0,
                'description': 'Error in firing range (max-min rate)',
            },
            'cv_isi_error': {
                'weight': 0.00,
                'min_val': 0.0,
                'max_val': 5.0,
                'description': 'Error in coefficient of variation of ISI',
            },
            'n_units_error': {
                'weight': 0.00,
                'min_val': 0.0,
                'max_val': 500.0,
                'description': 'Absolute difference in number of active units',
            },
            'ei_ratio_error': {
                'weight': 0.00,
                'min_val': 0.0,
                'max_val': 1.0,
                'description': 'Error in excitatory/inhibitory classification ratio',
            },
        },
    },

    # ------------------------------------------------------------------
    # 2. Quality / waveform-proxy metrics (from xlsx attrs)
    # ------------------------------------------------------------------
    'quality_metrics': {
        'weight': 0.00,
        'description': 'Quality metric distribution comparison (snr, amplitude, etc.)',
        'metrics': {
            'snr_error': {
                'weight': 0.30,
                'min_val': 0.0,
                'max_val': 20.0,
                'description': 'Error in mean SNR across units',
            },
            'amplitude_median_error': {
                'weight': 0.25,
                'min_val': 0.0,
                'max_val': 200.0,
                'description': 'Error in median amplitude',
            },
            'presence_ratio_error': {
                'weight': 0.20,
                'min_val': 0.0,
                'max_val': 1.0,
                'description': 'Error in mean presence ratio',
            },
            'isi_violations_error': {
                'weight': 0.25,
                'min_val': 0.0,
                'max_val': 1.0,
                'description': 'Error in ISI violation ratio',
            },
        },
    },

    # ------------------------------------------------------------------
    # 3. Synchrony metrics (computed from spike trains)
    # ------------------------------------------------------------------
    'synchrony': {
        'weight': 0.00,
        'description': 'Network-level synchronization comparison',
        'metrics': {
            'sync_spike_2_error': {
                'weight': 0.30,
                'min_val': 0.0,
                'max_val': 1.0,
                'description': 'Error in sync_spike_2 (2ms window)',
            },
            'sync_spike_4_error': {
                'weight': 0.30,
                'min_val': 0.0,
                'max_val': 1.0,
                'description': 'Error in sync_spike_4 (4ms window)',
            },
            'sync_spike_8_error': {
                'weight': 0.20,
                'min_val': 0.0,
                'max_val': 1.0,
                'description': 'Error in sync_spike_8 (8ms window)',
            },
            'pairwise_correlation_error': {
                'weight': 0.20,
                'min_val': 0.0,
                'max_val': 1.0,
                'description': 'Error in mean pairwise correlation',
            },
        },
    },

    # ------------------------------------------------------------------
    # 4-6. Hierarchical burst levels (parameter_free_burst_detector)
    #
    # NOTE on which metric names actually bite: the active scorer
    # (compute_hierarchical_burst_scores_simple_v2) reads exactly four keys --
    # burst_count_error, timing_error, duration_error (which drives the
    # peak-WIDTH term, a deliberate pre-existing remap), and burst_amp_error.
    # burst_rate_error / ibi_error / participation_error / intensity_error /
    # spikes_per_burst_error are inert; they are kept at 0.00 for documentation.
    # ------------------------------------------------------------------
    'burstlets': {
        'weight': 0.12,  # was 0.20 in schema_v3 (× 0.60)
        'description': 'Fast, small bursts (hierarchical level 1)',
        'metrics': {
            'burst_rate_error': {
                'weight': 0.00,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Error in burstlet rate (events/s) — inert in the active scorer',
            },
            'burst_count_error': {
                'weight': 0.40,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Normalized error in total burstlet count',
            },
            'duration_error': {
                'weight': 0.20,
                'min_val': 0.0,
                'max_val': 5.0,
                'description': 'Drives the burstlet peak-width term',
            },
            'burst_amp_error': {
                'weight': 0.15,
                'min_val': 0.0,
                'max_val': 10.0,
                'description': 'Error in mean per-unit burstlet peak (population rate at peak / n_units)',
            },
            'participation_error': {
                'weight': 0.0,
                'min_val': 0.0,
                'max_val': 1.0,
                'description': 'Error in mean unit participation fraction — inert',
            },
            'spikes_per_burst_error': {
                'weight': 0.0,
                'min_val': 0.0,
                'max_val': 500.0,
                'description': 'Error in mean spikes per burstlet — inert',
            },
            'timing_error': {
                'weight': 0.25,
                'min_val': 0.0,
                'max_val': 10.0,
                'description': 'Victor-Purpura timing distance on burstlet start times',
            },
        },
    },

    'network_bursts': {
        'weight': 0.06,  # was 0.10 in schema_v3 (× 0.60)
        'description': 'Merged medium-scale bursts (hierarchical level 2)',
        'metrics': {
            'burst_rate_error': {
                'weight': 0.00,
                'min_val': 0.0,
                'max_val': 50.0,
                'description': 'Error in network burst rate (events/s) — inert in the active scorer',
            },
            'burst_count_error': {
                'weight': 0.40,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Normalized error in network burst count',
            },
            'duration_error': {
                'weight': 0.20,
                'min_val': 0.0,
                'max_val': 10.0,
                'description': 'Drives the network-burst peak-width term',
            },
            'burst_amp_error': {
                'weight': 0.15,
                'min_val': 0.0,
                'max_val': 10.0,
                'description': 'Error in mean per-unit network-burst peak',
            },
            'ibi_error': {
                'weight': 0.00,
                'min_val': 0.0,
                'max_val': 20.0,
                'description': 'Error in inter-burst interval — inert',
            },
            'participation_error': {
                'weight': 0.00,
                'min_val': 0.0,
                'max_val': 1.0,
                'description': 'Error in unit participation fraction — inert',
            },
            'intensity_error': {
                'weight': 0.00,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Error in synchrony energy / intensity — inert',
            },
            'timing_error': {
                'weight': 0.25,
                'min_val': 0.0,
                'max_val': 10.0,
                'description': 'Victor-Purpura timing distance on network-burst start times',
            },
        },
    },

    'superbursts': {
        'weight': 0.00,
        'description': 'Long, clustered burst events (hierarchical level 3)',
        'metrics': {
            'burst_rate_error': {
                'weight': 0.25,
                'min_val': 0.0,
                'max_val': 10.0,
                'description': 'Error in superburst rate (events/s) — inert',
            },
            'burst_count_error': {
                'weight': 0.25,
                'min_val': 0.0,
                'max_val': 50.0,
                'description': 'Normalized error in superburst count',
            },
            'duration_error': {
                'weight': 0.20,
                'min_val': 0.0,
                'max_val': 20.0,
                'description': 'Drives the superburst peak-width term',
            },
            'burst_amp_error': {
                'weight': 0.15,
                'min_val': 0.0,
                'max_val': 10.0,
                'description': 'Error in mean per-unit superburst peak',
            },
            'timing_error': {
                'weight': 0.15,
                'min_val': 0.0,
                'max_val': 5.0,
                'description': 'Victor-Purpura timing distance on superburst start times',
            },
        },
    },

    'pre_burstlets': {
        'weight': 0.12,  # was 0.20 in schema_v3 (× 0.60)
        'description': 'Pre-detection burstlets (using permissive threshold)',
        'metrics': {
            'burst_rate_error': {
                'weight': 0.00,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Error in burstlet rate (events/s) — inert',
            },
            'burst_count_error': {
                'weight': 0.40,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Normalized error in total burstlet count',
            },
            'duration_error': {
                'weight': 0.20,
                'min_val': 0.0,
                'max_val': 5.0,
                'description': 'Drives the pre-burstlet peak-width term',
            },
            'burst_amp_error': {
                'weight': 0.15,
                'min_val': 0.0,
                'max_val': 10.0,
                'description': 'Error in mean per-unit pre-burstlet peak',
            },
            'participation_error': {
                'weight': 0.0,
                'min_val': 0.0,
                'max_val': 1.0,
                'description': 'Error in mean unit participation fraction — inert',
            },
            'spikes_per_burst_error': {
                'weight': 0.0,
                'min_val': 0.0,
                'max_val': 500.0,
                'description': 'Error in mean spikes per burstlet — inert',
            },
            'timing_error': {
                'weight': 0.25,
                'min_val': 0.0,
                'max_val': 10.0,
                'description': 'Victor-Purpura timing distance on burstlet start times',
            },
        },
    },

    # ------------------------------------------------------------------
    # 7. Drug-response ratios vs the THEORETICAL target
    #
    # Each metric name must appear in fitnessFunc_v2._DRUG_RATIO_FEATURE_MAP AND
    # as a dataset in /drug_effects/<drug>/ of the target h5. A metric listed here
    # but missing from the h5 is silently skipped at fitnessFunc_v2.py:~1853,
    # which quietly shrinks the effective drug weight rather than erroring.
    #
    # Excluded on purpose: superburst_* (target ratio is 0.0 experimentally, no
    # gradient) and pre_burstlet_* (redundant with the burstlet level).
    # ------------------------------------------------------------------
    'drug_response': {
        'weight': 0.40,
        'description': 'Simulated post/pre ratios vs THEORETICAL drug ratios',
        # Log-space, softly-saturating per-feature loss instead of the legacy
        # min(|sim-exp|/max_dev, 1.0). The clipped form has a dead zone that
        # measurably destroyed the drug gradient: in the smoke_v4 batch, 81 of 99
        # trials scored the bicuculline term at exactly 24899.958, because 91 of
        # 99 produced zero detected bursts under GABA block and every ratio from
        # 0 to 1 clips to the same loss. 'log_soft' is strictly monotone, so
        # "fewer bursts" and "no bursts" are finally distinguishable.
        # schema_v3_drug.py omits this key and keeps the legacy 'clipped' loss,
        # so experimental-target runs are unaffected.
        'loss': 'log_soft',
        'metrics': {
            'network_burst_duration_ratio': {
                'weight': 0.20,
                'max_dev': 2.0,   # theory ratio 3.0
                'description': 'Network burst duration ratio (post/pre). The headline '
                               'disinhibition effect: no GABA-A brake to terminate a burst.',
            },
            'pop_FR_ratio': {
                'weight': 0.15,
                'max_dev': 1.0,   # theory ratio 2.0
                'description': 'Population firing-rate ratio (post/pre)',
            },
            'network_burst_rate_ratio': {
                'weight': 0.15,
                'max_dev': 0.8,   # theory ratio 1.8
                'description': 'Network burst rate ratio (post/pre); detector static=40',
            },
            'network_burst_amp_ratio': {
                'weight': 0.15,
                'max_dev': 1.5,   # theory ratio 2.5
                'description': 'Network burst amplitude ratio (post/pre) — mean per-unit '
                               'burst peak; measures how hard each burst recruits',
            },
            'exc_firing_rate_ratio': {
                'weight': 0.10,
                'max_dev': 1.2,   # theory ratio 2.2
                'description': 'Excitatory mean firing-rate ratio; E/I split is the '
                               'top-20%-by-baseline-rate proxy, matching the experimental target',
            },
            'inh_firing_rate_ratio': {
                'weight': 0.10,
                'max_dev': 0.5,   # theory ratio 1.5
                'description': 'Inhibitory mean firing-rate ratio (same proxy split)',
            },
            'burstlet_duration_ratio': {
                'weight': 0.10,
                'max_dev': 1.0,   # theory ratio 2.0
                'description': 'Burstlet duration ratio (post/pre); detector static=40',
            },
            'burstlet_amp_ratio': {
                'weight': 0.05,
                'max_dev': 1.0,   # theory ratio 2.0
                'description': 'Burstlet amplitude ratio (post/pre)',
            },
        },
    },
}


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    # parameter_free_burst_detector parameters
    'hierarchical_bursts': {
        'gamma': 1.0,
        'min_burstlet_participation': 0.20,
        'min_absolute_rate_Hz': 0.5,
        'min_burst_density_Hz': 1.0,
        'min_relative_height': 0.1,
        'extent_frac': 0.30,
        'bin_ms': 10,
        'smoothing_min_ms': 20,
        'burstlet_merge_gap_s': 0.1,
        'network_merge_gap_s': 1.0,
        'superburst_min_duration_s': 2.5,
    },
    'unit': {
        'bin_size': 0.01,   # for rate histograms
    },
    'synchrony': {
        'bin_size': 0.01,   # for pairwise correlation
    },
    'recording_duration': 300.0,   # seconds (overridden from data)
    'max_fitness': 1000.0,
}


# =============================================================================
# EXPERIMENTAL HDF5 ATTRIBUTE LIST (for reference / validation)
# =============================================================================

EXPERIMENTAL_UNIT_ATTRS = [
    'num_spikes', 'firing_rate', 'presence_ratio', 'snr',
    'isi_violations_ratio', 'isi_violations_count',
    'rp_contamination', 'rp_violations', 'sliding_rp_violation',
    'amplitude_cutoff', 'amplitude_median',
    'amplitude_cv_median', 'amplitude_cv_range',
    'sync_spike_2', 'sync_spike_4', 'sync_spike_8',
    'firing_range', 'sd_ratio',
    'noise_cutoff', 'noise_ratio',
    'loc_x', 'loc_y',
    'cell_type',
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_component_weights() -> Dict[str, float]:
    """Return top-level component weights from the schema."""
    return {name: comp['weight'] for name, comp in fit_schema.items()}


def get_metric_weights(component: str) -> Dict[str, float]:
    """Return metric weights for a specific component."""
    if component not in fit_schema:
        raise ValueError(f"Unknown component: {component}")
    return {
        name: m['weight']
        for name, m in fit_schema[component]['metrics'].items()
    }


def normalize_schema_weights() -> Dict:
    """Return a copy of fit_schema with all weight groups normalised to 1.0."""
    normalized = copy.deepcopy(fit_schema)

    # top-level
    total = sum(c['weight'] for c in normalized.values())
    if total > 0:
        for c in normalized.values():
            c['weight'] /= total

    # per-component metric weights
    for component in normalized.values():
        mtotal = sum(m['weight'] for m in component['metrics'].values())
        if mtotal > 0:
            for m in component['metrics'].values():
                m['weight'] /= mtotal

    return normalized


def validate_weights() -> bool:
    """Check that all weight groups sum to ~1.0, and that the 40/60 split holds."""
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

    drug_w = fit_schema['drug_response']['weight']
    baseline_w = sum(c['weight'] for n, c in fit_schema.items() if n != 'drug_response')
    if not np.isclose(drug_w, 0.40):
        print(f"Warning: drug_response weight is {drug_w:.4f}, expected 0.40")
        ok = False
    if not np.isclose(baseline_w, 0.60):
        print(f"Warning: baseline components sum to {baseline_w:.4f}, expected 0.60")
        ok = False
    return ok


def get_scoring_bounds(component: str, metric: str) -> Dict[str, float]:
    """Get min/max bounds for a specific metric (drug_response uses max_dev)."""
    if component not in fit_schema:
        raise ValueError(f"Unknown component: {component}")
    if metric not in fit_schema[component]['metrics']:
        raise ValueError(f"Unknown metric {metric} in {component}")
    m = fit_schema[component]['metrics'][metric]
    if component == 'drug_response':
        return {'max_dev': m['max_dev']}
    return {'min_val': m['min_val'], 'max_val': m['max_val']}


def get_full_config() -> Dict:
    """Return schema + config combined."""
    return {
        'schema': fit_schema,
        'config': DEFAULT_CONFIG,
        'unit_attrs': EXPERIMENTAL_UNIT_ATTRS,
    }


# =============================================================================
if __name__ == "__main__":
    print("=" * 68)
    print("FITNESS SCHEMA v3 (THEORETICAL drug target) SUMMARY")
    print("=" * 68)

    for cname, comp in fit_schema.items():
        print(f"\n{cname.upper()} (weight={comp['weight']:.2f})")
        print(f"  {comp['description']}")
        for mname, m in comp['metrics'].items():
            bounds = (f"max_dev={m['max_dev']:.1f}"
                      if cname == 'drug_response'
                      else f"[{m['min_val']:.1f}, {m['max_val']:.1f}]")
            print(f"    {mname:32s} w={m['weight']:.2f}  {bounds}")

    _drug = fit_schema['drug_response']['weight']
    _base = sum(c['weight'] for n, c in fit_schema.items() if n != 'drug_response')
    print(f"\nSplit: drug_response={_drug:.2f}  baseline={_base:.2f}")
    print(f"Weights valid: {validate_weights()}")
    print("✓ Schema module loaded.")
