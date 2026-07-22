"""
Fitness Schema for Claude-based Network Optimization
======================================================

Defines the fitness calculation schema that maps between:
- The new experimental_data.h5 format (from data_processing.process_kilosort)
- The metrics needed for fitness calculation
- Weight configurations

This schema is designed to work with fitnessFunc_claude.py and the
experimental HDF5 data format produced by the new data processing pipeline.

Author: Claude Implementation
Date: January 2026
"""

from typing import Any, Dict, Optional

import numpy as np


# =============================================================================
# FITNESS SCHEMA DEFINITION
# =============================================================================

fit_schema = {
    'spike_timing': {
        'weight': 0.0,
        'description': 'Spike timing precision metrics using distance measures',
        'metrics': {
            'victor_purpura_distance': {
                'weight': 0.5,
                'min_val': 0.0,
                'max_val': 1000.0,
                'description': 'Victor-Purpura spike train distance (lower is better)',
            },
            'mean_distance': {
                'weight': 0.3,
                'min_val': 0.0,
                'max_val': 500.0,
                'description': 'Mean spike timing distance across units',
            },
            'van_rossum_distance': {
                'weight': 0.2,
                'min_val': 0.0,
                'max_val': 500.0,
                'description': 'van Rossum spike train distance',
            },
        },
    },
    'burst_metrics': {
        'weight': 0.00,
        'description': 'Burst characteristics comparison',
        'metrics': {
            'burst_rate_error': {
                'weight': 0.35,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Error in burst rate (bursts/second)',
            },
            'burst_duration_error': {
                'weight': 0.25,
                'min_val': 0.0,
                'max_val': 1.0,
                'description': 'Error in mean burst duration (seconds)',
            },
            'ibi_error': {
                'weight': 0.25,
                'min_val': 0.0,
                'max_val': 10.0,
                'description': 'Error in inter-burst interval',
            },
            'spikes_per_burst_error': {
                'weight': 0.15,
                'min_val': 0.0,
                'max_val': 50.0,
                'description': 'Error in mean spikes per burst',
            },
        },
    },
    'unit_metrics': {
        'weight': 0.0,
        'description': 'Single-unit firing property metrics',
        'metrics': {
            'firing_rate_mae': {
                'weight': 0.40,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Mean absolute error in firing rates (Hz)',
            },
            'cv_isi_error': {
                'weight': 0.30,
                'min_val': 0.0,
                'max_val': 5.0,
                'description': 'Error in coefficient of variation of ISI',
            },
            'spike_count_error': {
                'weight': 0.20,
                'min_val': 0.0,
                'max_val': 1000.0,
                'description': 'Error in total spike counts',
            },
            'isi_distribution_error': {
                'weight': 0.10,
                'min_val': 0.0,
                'max_val': 1.0,
                'description': 'KS-statistic for ISI distribution comparison',
            },
        },
    },
    'synchrony': {
        'weight': 0.00,
        'description': 'Network-level synchronization metrics',
        'metrics': {
            'global_synchrony_error': {
                'weight': 0.40,
                'min_val': 0.0,
                'max_val': 1.0,
                'description': 'Error in global synchrony index',
            },
            'correlation_matrix_error': {
                'weight': 0.35,
                'min_val': 0.0,
                'max_val': 10.0,
                'description': 'Frobenius norm of correlation matrix difference',
            },
            'pairwise_correlation_error': {
                'weight': 0.25,
                'min_val': 0.0,
                'max_val': 1.0,
                'description': 'Error in mean pairwise correlation',
            },
        },
    },
    'burstlets': {
        'weight': 01.00,
        'description': 'Fast, small bursts (hierarchical level 1)',
        'metrics': {
            'burst_rate_error': {
                'weight': 0.50,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Error in burstlet rate (bursts/second)',
            },
            'burst_count_error': {
                'weight': 0.00,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Normalized error in total burstlet count',
            },
            'duration_error': {
                'weight': 0.50,
                'min_val': 0.0,
                'max_val': 5.0,
                'description': 'Error in mean burstlet duration (seconds)',
            },
        },
    },
    'network_bursts': {
        'weight': 0.00,
        'description': 'Merged medium-scale bursts (hierarchical level 2)',
        'metrics': {
            'burst_rate_error': {
                'weight': 0.25,
                'min_val': 0.0,
                'max_val': 50.0,
                'description': 'Error in network burst rate (bursts/second)',
            },
            'burst_count_error': {
                'weight': 0.20,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Normalized error in total network burst count',
            },
            'duration_error': {
                'weight': 0.20,
                'min_val': 0.0,
                'max_val': 10.0,
                'description': 'Error in mean network burst duration (seconds)',
            },
            'ibi_error': {
                'weight': 0.15,
                'min_val': 0.0,
                'max_val': 20.0,
                'description': 'Error in inter-burst interval for network bursts',
            },
            'participation_error': {
                'weight': 0.10,
                'min_val': 0.0,
                'max_val': 1.0,
                'description': 'Error in unit participation fraction',
            },
            'synchrony_energy_error': {
                'weight': 0.10,
                'min_val': 0.0,
                'max_val': 100.0,
                'description': 'Error in synchrony energy (integral of synchrony signal)',
            },
        },
    },
    'superbursts': {
        'weight': 0.00,
        'description': 'Long, clustered burst events (hierarchical level 3)',
        'metrics': {
            'burst_rate_error': {
                'weight': 0.40,
                'min_val': 0.0,
                'max_val': 10.0,
                'description': 'Error in superburst rate (bursts/second)',
            },
            'burst_count_error': {
                'weight': 0.30,
                'min_val': 0.0,
                'max_val': 50.0,
                'description': 'Normalized error in total superburst count',
            },
            'duration_error': {
                'weight': 0.30,
                'min_val': 0.0,
                'max_val': 20.0,
                'description': 'Error in mean superburst duration (seconds)',
            },
        },
    },
}


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    'spike_timing': {
        'method': 'victor_purpura',
        'q_values': [0.1, 0.5, 1.0, 2.0],  # VP temporal precision parameters
        'tau': 0.01,  # van Rossum time constant (seconds)
    },
    'burst': {
        'isi_threshold': 0.1,  # ISI threshold for burst detection (seconds)
        'min_spikes': 3,  # Minimum spikes to constitute a burst
    },
    'unit': {
        'bin_size': 0.01,  # Bin size for rate calculations (seconds)
    },
    'synchrony': {
        'bin_size': 0.01,  # Bin size for correlation computation (seconds)
    },
    'hierarchical_bursts': {
        # Adaptive burst detection parameters (used for all levels)
        'gamma': 2.0,  # Power for synchrony signal P^gamma
        'min_burstlet_participation': 0.20,  # Minimum fraction of units (20%)
        'min_absolute_rate_Hz': 0.5,  # Minimum absolute firing rate
        'min_burst_density_Hz': 1.0,  # Minimum burst density
        'min_relative_height': 0.25,  # Minimum peak height (25% of global max)
        'bin_ms': 10.0,  # Maximum bin size in milliseconds
        'smoothing_min_ms': 20.0,  # Minimum smoothing window
        'superburst_min_duration_s': 2.5,  # Minimum superburst duration
    },
    'recording_duration': 300.0,  # Default recording duration (seconds)
    'max_fitness': 1000.0,  # Maximum fitness score (worst case)
}


# =============================================================================
# EXPERIMENTAL DATA MAPPING
# =============================================================================

# Maps HDF5 keys to internal representation
EXPERIMENTAL_DATA_MAP = {
    # Direct spike data
    'spike_times': 'spike_times',
    'spike_clusters': 'spike_clusters',
    'unit_ids': 'unit_ids',
    
    # Pre-computed metrics
    'metrics/firing_rates': 'firing_rates',
    'metrics/cv_isi': 'cv_isi',
    'metrics/isi_mean': 'isi_mean',
    'metrics/isi_std': 'isi_std',
    'metrics/isi_median': 'isi_median',
    'metrics/spike_counts': 'spike_counts',
    'metrics/refractory_violations': 'refractory_violations',
    'metrics/refractory_violation_rate': 'refractory_violation_rate',
    'metrics/unit_ids': 'metrics_unit_ids',
    
    # Synchrony data
    'synchrony/mean_pairwise_correlation': 'mean_pairwise_correlation',
    'synchrony/synchrony_index': 'synchrony_index',
    'synchrony/population_rate': 'population_rate',
    'synchrony/bin_centers': 'synchrony_bin_centers',
    'synchrony/bin_size': 'synchrony_bin_size',
    'synchrony/network_burst_rate': 'network_burst_rate',
    'synchrony/num_network_bursts': 'num_network_bursts',
    
    # Metadata (JSON encoded)
    'bursts.JSON': 'bursts_json',
    'meta.JSON': 'meta_json',
    'synchrony/network_bursts.JSON': 'network_bursts_json',
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_component_weights() -> Dict[str, float]:
    """
    Get the top-level component weights from the schema.
    
    Returns
    -------
    weights : dict
        {component_name: weight}
    """
    return {name: component['weight'] for name, component in fit_schema.items()}


def get_metric_weights(component: str) -> Dict[str, float]:
    """
    Get metric weights for a specific component.
    
    Parameters
    ----------
    component : str
        Component name ('spike_timing', 'burst_metrics', 'unit_metrics', 'synchrony')
    
    Returns
    -------
    weights : dict
        {metric_name: weight}
    """
    if component not in fit_schema:
        raise ValueError(f"Unknown component: {component}")
    
    return {
        name: metric['weight'] 
        for name, metric in fit_schema[component]['metrics'].items()
    }


def validate_weights() -> bool:
    """
    Validate that all weights sum to 1.0.
    
    Returns
    -------
    valid : bool
        True if all weights are valid
    """
    # Check top-level weights
    total_weight = sum(comp['weight'] for comp in fit_schema.values())
    if not np.isclose(total_weight, 1.0):
        print(f"Warning: Component weights sum to {total_weight}, not 1.0")
        return False
    
    # Check metric weights within each component
    for comp_name, component in fit_schema.items():
        metric_weight = sum(m['weight'] for m in component['metrics'].values())
        if not np.isclose(metric_weight, 1.0):
            print(f"Warning: {comp_name} metric weights sum to {metric_weight}, not 1.0")
            return False
    
    return True


def normalize_schema_weights() -> Dict:
    """
    Return a copy of fit_schema with normalized weights.
    
    Returns
    -------
    normalized_schema : dict
        Schema with normalized weights
    """
    import copy
    normalized = copy.deepcopy(fit_schema)
    
    # Normalize top-level weights
    total_weight = sum(comp['weight'] for comp in normalized.values())
    for comp in normalized.values():
        comp['weight'] /= total_weight
    
    # Normalize metric weights within each component
    for component in normalized.values():
        metric_total = sum(m['weight'] for m in component['metrics'].values())
        for metric in component['metrics'].values():
            metric['weight'] /= metric_total
    
    return normalized


def get_scoring_bounds(component: str, metric: str) -> Dict[str, float]:
    """
    Get the scoring bounds for a specific metric.
    
    Parameters
    ----------
    component : str
        Component name
    metric : str
        Metric name
    
    Returns
    -------
    bounds : dict
        {'min_val': float, 'max_val': float}
    """
    if component not in fit_schema:
        raise ValueError(f"Unknown component: {component}")
    
    if metric not in fit_schema[component]['metrics']:
        raise ValueError(f"Unknown metric {metric} in component {component}")
    
    metric_spec = fit_schema[component]['metrics'][metric]
    return {
        'min_val': metric_spec['min_val'],
        'max_val': metric_spec['max_val'],
    }


def get_full_config() -> Dict:
    """
    Get full configuration including schema and default config.
    
    Returns
    -------
    config : dict
        Complete configuration dictionary
    """
    return {
        'schema': fit_schema,
        'config': DEFAULT_CONFIG,
        'data_map': EXPERIMENTAL_DATA_MAP,
    }


# =============================================================================
# SCHEMA FOR LEGACY COMPATIBILITY
# =============================================================================

# Schema format compatible with existing fitnessFunc.py structure
LEGACY_COMPATIBLE_SCHEMA = {
    'spiking_data': {
        'spike_times': False,
        'spiking_times_by_unit': False,
        'spiking_metrics_by_unit': False,
        'frs': {'include': True, 'weight': 1.0, 'normalize': True},
        'i_frs': {'include': True, 'weight': 1.0, 'normalize': True},
        'e_frs': {'include': True, 'weight': 1.0, 'normalize': True},
        'u_frs': False,
        'EI_fr_ratios': {'include': True, 'weight': 1.0, 'normalize': True},
        'isi': {'include': True, 'weight': 1.0, 'normalize': True},
        'i_isi': {'include': True, 'weight': 1.0, 'normalize': True},
        'e_isi': {'include': True, 'weight': 1.0, 'normalize': True},
        'u_isi': False,
    },
    'bursting_data': False,
    'mega_bursting_data': {
        'ax': False,
        'convolved_data': False,
        'unit_metrics': False,
        'burst_metrics': {
            'num_bursts': False,
            'burst_rate': {'include': True, 'weight': 1.5, 'normalize': True},
            'burst_ids': False,
            'ibi': {'include': True, 'weight': 1.0, 'normalize': True},
            'burst_amp': {'include': True, 'weight': 1.0, 'normalize': True},
            'burst_duration': {'include': True, 'weight': 1.0, 'normalize': True},
            'burst_parts': False,
            'num_units_per_burst': {'include': True, 'weight': 1.0, 'normalize': True},
            'in_burst_fr': {'include': True, 'weight': 1.0, 'normalize': True},
        },
        'warnings': False,
    },
    'HFBursting_metrics': False,
    'unit_types': False,
    'unit_locations': False,
    'simData': False,
    'popData': False,
    'cellData': False,
}


if __name__ == "__main__":
    # Test schema validation
    print("Testing Fitness Schema...")
    print(f"\nComponent Weights: {get_component_weights()}")
    print(f"Weights Valid: {validate_weights()}")
    
    print("\n" + "=" * 60)
    print("FITNESS SCHEMA SUMMARY")
    print("=" * 60)
    
    for comp_name, component in fit_schema.items():
        print(f"\n{comp_name.upper()} (weight={component['weight']:.2f})")
        print(f"  {component['description']}")
        print("  Metrics:")
        for metric_name, metric in component['metrics'].items():
            print(f"    - {metric_name}: weight={metric['weight']:.2f}, "
                  f"bounds=[{metric['min_val']}, {metric['max_val']}]")
    
    print("\n✓ Schema module working!")
