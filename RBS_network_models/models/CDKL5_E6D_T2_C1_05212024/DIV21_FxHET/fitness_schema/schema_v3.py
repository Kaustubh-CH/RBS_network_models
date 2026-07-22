"""
Fitness Schema v3 === More weightage to Inhibitory firing rate
==========================================

Schema aligned with experimental_features.h5 produced by create_experimental_target.py.

Experimental HDF5 structure (per unit):
    unit_{id}/spike_times         — dataset
    unit_{id}.attrs               — num_spikes, firing_rate, presence_ratio, snr,
                                    isi_violations_ratio, isi_violations_count,
                                    rp_contamination, rp_violations,
                                    sliding_rp_violation, amplitude_cutoff,
                                    amplitude_median, amplitude_cv_median,
                                    amplitude_cv_range, sync_spike_2,
                                    sync_spike_4, sync_spike_8, firing_range,
                                    sd_ratio, noise_cutoff, noise_ratio,
                                    loc_x, loc_y, cell_type

Network burst features come from parameter_free_burst_detector.compute_network_bursts
which yields three hierarchical levels:
    burstlets, network_bursts, superbursts
each with: count, rate, duration, inter_event_interval, intensity, participation,
           spikes_per_burst, burst_peak

Author: Auto-generated
Date: February 2026
"""

import copy
import numpy as np
from typing import Any, Dict, Optional


# =============================================================================
# FITNESS SCHEMA — maps directly to what we measure
# =============================================================================

fit_schema = {
    # ------------------------------------------------------------------
    # 1. Per-unit spiking metrics (from xlsx / computed on-the-fly)
    # ------------------------------------------------------------------
    'unit_metrics': {
        'weight': 0.50,
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
    # ------------------------------------------------------------------
    'burstlets': {
        'weight': 0.20,
        'description': 'Fast, small bursts (hierarchical level 1)',
        'metrics': {
            'burst_rate_error': {
                'weight': 0.00,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Error in burstlet rate (events/s)',
            },
            'burst_count_error': {
                'weight': 0.45,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Normalized error in total burstlet count',
            },
            'duration_error': {
                'weight': 0.25,
                'min_val': 0.0,
                'max_val': 5.0,
                'description': 'Error in mean burstlet duration (s)',
            },
            'participation_error': {
                'weight': 0.0,
                'min_val': 0.0,
                'max_val': 1.0,
                'description': 'Error in mean unit participation fraction',
            },
            'spikes_per_burst_error': {
                'weight': 0.0,
                'min_val': 0.0,
                'max_val': 500.0,
                'description': 'Error in mean spikes per burstlet',
            },
            'timing_error': {
                'weight': 0.30,
                'min_val': 0.0,
                'max_val': 10.0,
                'description': 'Victor-Purpura timing distance on burstlet start times',
            },
        },
    },

    'network_bursts': {
        'weight': 0.10,
        'description': 'Merged medium-scale bursts (hierarchical level 2)',
        'metrics': {
            'burst_rate_error': {
                'weight': 0.00,
                'min_val': 0.0,
                'max_val': 50.0,
                'description': 'Error in network burst rate (events/s)',
            },
            'burst_count_error': {
                'weight': 0.45,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Normalized error in network burst count',
            },
            'duration_error': {
                'weight': 0.25,
                'min_val': 0.0,
                'max_val': 10.0,
                'description': 'Error in mean network burst duration (s)',
            },
            'ibi_error': {
                'weight': 0.00,
                'min_val': 0.0,
                'max_val': 20.0,
                'description': 'Error in inter-burst interval',
            },
            'participation_error': {
                'weight': 0.00,
                'min_val': 0.0,
                'max_val': 1.0,
                'description': 'Error in unit participation fraction',
            },
            'intensity_error': {
                'weight': 0.00,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Error in synchrony energy / intensity',
            },
            'timing_error': {
                'weight': 0.30,
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
                'weight': 0.30,
                'min_val': 0.0,
                'max_val': 10.0,
                'description': 'Error in superburst rate (events/s)',
            },
            'burst_count_error': {
                'weight': 0.25,
                'min_val': 0.0,
                'max_val': 50.0,
                'description': 'Normalized error in superburst count',
            },
            'duration_error': {
                'weight': 0.25,
                'min_val': 0.0,
                'max_val': 20.0,
                'description': 'Error in mean superburst duration (s)',
            },
            'timing_error': {
                'weight': 0.20,
                'min_val': 0.0,
                'max_val': 5.0,
                'description': 'Victor-Purpura timing distance on superburst start times',
            },
        },
    },

    'pre_burstlets': {
        'weight': 0.20,
        'description': 'Pre-detection burstlets (using permissive threshold)',
        'metrics': {
            'burst_rate_error': {
                'weight': 0.00,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Error in burstlet rate (events/s)',
            },
            'burst_count_error': {
                'weight': 0.45,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Normalized error in total burstlet count',
            },
            'duration_error': {
                'weight': 0.25,
                'min_val': 0.0,
                'max_val': 5.0,
                'description': 'Error in mean burstlet duration (s)',
            },
            'participation_error': {
                'weight': 0.0,
                'min_val': 0.0,
                'max_val': 1.0,
                'description': 'Error in mean unit participation fraction',
            },
            'spikes_per_burst_error': {
                'weight': 0.0,
                'min_val': 0.0,
                'max_val': 500.0,
                'description': 'Error in mean spikes per burstlet',
            },
            'timing_error': {
                'weight': 0.30,
                'min_val': 0.0,
                'max_val': 10.0,
                'description': 'Victor-Purpura timing distance on burstlet start times',
            },
        },
    }
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


def get_scoring_bounds(component: str, metric: str) -> Dict[str, float]:
    """Get min/max bounds for a specific metric."""
    if component not in fit_schema:
        raise ValueError(f"Unknown component: {component}")
    if metric not in fit_schema[component]['metrics']:
        raise ValueError(f"Unknown metric {metric} in {component}")
    m = fit_schema[component]['metrics'][metric]
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
    print("=" * 60)
    print("FITNESS SCHEMA v2 SUMMARY")
    print("=" * 60)

    for cname, comp in fit_schema.items():
        print(f"\n{cname.upper()} (weight={comp['weight']:.2f})")
        print(f"  {comp['description']}")
        for mname, m in comp['metrics'].items():
            print(f"    {mname:30s} w={m['weight']:.2f}  [{m['min_val']:.1f}, {m['max_val']:.1f}]")

    print(f"\nWeights valid: {validate_weights()}")
    print("✓ Schema module loaded.")
