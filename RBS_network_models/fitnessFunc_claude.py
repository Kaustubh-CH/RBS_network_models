"""
Fitness Function for Claude-based Network Optimization
=======================================================

A fitness function module that integrates with the existing RBS_network_models
framework but uses the experimental data format from the new data processing pipeline.

This module provides:
- Loading experimental data from HDF5 files (from data_processing.process_kilosort)
- Multi-metric fitness computation (spike timing, burst, unit, synchrony)
- Compatible return structure with fitnessFunc_v3 for batch optimization

Author: Claude Implementation
Date: January 2026
"""

import json
import logging
import os
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

# Import netpyne for simulation data loading
try:
    from netpyne import sim
except ImportError:
    sim = None
    logging.warning("netpyne not available - some features may not work")

# Import compute_network_metrics for detailed metrics
try:
    from MEA_Analysis.NetworkAnalysis.awNetworkAnalysis.network_analysis import compute_network_metrics
except ImportError:
    compute_network_metrics = None
    logging.warning("compute_network_metrics not available - using basic metrics")

# Handle both relative and absolute imports
try:
    from fitness_schema.schema_claude import (
        DEFAULT_CONFIG,
        fit_schema,
        get_component_weights,
        normalize_schema_weights,
    )
except ImportError:
    from fitness_schema.schema_claude import (
        DEFAULT_CONFIG,
        fit_schema,
        get_component_weights,
        normalize_schema_weights,
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# EXPERIMENTAL DATA LOADING
# =============================================================================

def load_experimental_h5(h5_path: str) -> Dict[str, Any]:
    """
    Load experimental data from HDF5 file produced by data_processing.process_kilosort.
    
    Parameters
    ----------
    h5_path : str
        Path to experimental_data.h5 file
    
    Returns
    -------
    exp_data : dict
        Extracted experimental data ready for fitness computation
    """
    import h5py
    
    exp_data = {
        'spike_data': {},
        'metrics': {},
        'synchrony': {},
        'bursts': [],
        'metadata': {},
    }
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Load spike times and clusters
            spike_times = f['spike_times'][:]  # in seconds
            spike_clusters = f['spike_clusters'][:]
            unit_ids = f['unit_ids'][:]
            
            # Build spike_data dict: {unit_id: spike_times_array}
            for unit_id in unit_ids:
                mask = spike_clusters == unit_id
                exp_data['spike_data'][int(unit_id)] = spike_times[mask]
            
            # Load pre-computed metrics
            if 'metrics' in f:
                metrics_group = f['metrics']
                exp_data['metrics']['firing_rates'] = metrics_group['firing_rates'][:]
                exp_data['metrics']['cv_isi'] = metrics_group['cv_isi'][:]
                exp_data['metrics']['isi_mean'] = metrics_group['isi_mean'][:]
                exp_data['metrics']['isi_std'] = metrics_group['isi_std'][:]
                exp_data['metrics']['isi_median'] = metrics_group['isi_median'][:]
                exp_data['metrics']['spike_counts'] = metrics_group['spike_counts'][:]
                exp_data['metrics']['unit_ids'] = metrics_group['unit_ids'][:]
            
            # Load synchrony data
            if 'synchrony' in f:
                sync_group = f['synchrony']
                exp_data['synchrony']['mean_pairwise_correlation'] = float(
                    sync_group['mean_pairwise_correlation'][0]
                )
                exp_data['synchrony']['synchrony_index'] = int(
                    sync_group['synchrony_index'][0]
                )
                exp_data['synchrony']['population_rate'] = sync_group['population_rate'][:]
                exp_data['synchrony']['bin_centers'] = sync_group['bin_centers'][:]
                exp_data['synchrony']['bin_size'] = float(sync_group['bin_size'][0])
                
                if 'network_burst_rate' in sync_group:
                    exp_data['synchrony']['network_burst_rate'] = float(
                        sync_group['network_burst_rate'][0]
                    )
                if 'num_network_bursts' in sync_group:
                    exp_data['synchrony']['num_network_bursts'] = int(
                        sync_group['num_network_bursts'][0]
                    )
                
                # Load network bursts JSON if present (legacy)
                if 'network_bursts.JSON' in sync_group:
                    try:
                        network_bursts_str = sync_group['network_bursts.JSON'][0]
                        if isinstance(network_bursts_str, bytes):
                            network_bursts_str = network_bursts_str.decode('utf-8')
                        exp_data['synchrony']['network_bursts'] = json.loads(network_bursts_str)
                    except Exception as e:
                        logger.warning(f"Could not parse network_bursts.JSON: {e}")
                
                # Load hierarchical bursts JSON if present (new)
                if 'hierarchical_bursts.JSON' in sync_group:
                    try:
                        hierarchical_bursts_str = sync_group['hierarchical_bursts.JSON'][0]
                        if isinstance(hierarchical_bursts_str, bytes):
                            hierarchical_bursts_str = hierarchical_bursts_str.decode('utf-8')
                        hierarchical_data = json.loads(hierarchical_bursts_str)
                        
                        # Store as bursting_data_v2 for compatibility with fitness function
                        exp_data['bursting_data_v2'] = {
                            'burstlets': hierarchical_data.get('burstlets', []),
                            'network_bursts': hierarchical_data.get('network_bursts', []),
                            'superbursts': hierarchical_data.get('superbursts', []),
                            'diagnostics': hierarchical_data.get('diagnostics', {}),
                            'baseline': hierarchical_data.get('baseline', 0.0),
                        }
                        logger.info(f"Loaded hierarchical bursts: {len(hierarchical_data.get('burstlets', []))} burstlets, "
                                  f"{len(hierarchical_data.get('network_bursts', []))} network bursts, "
                                  f"{len(hierarchical_data.get('superbursts', []))} superbursts")
                    except Exception as e:
                        logger.warning(f"Could not parse hierarchical_bursts.JSON: {e}")
            
            # Load burst data JSON if present
            if 'bursts.JSON' in f:
                try:
                    bursts_str = f['bursts.JSON'][0]
                    if isinstance(bursts_str, bytes):
                        bursts_str = bursts_str.decode('utf-8')
                    exp_data['bursts'] = json.loads(bursts_str)
                except Exception as e:
                    logger.warning(f"Could not parse bursts.JSON: {e}")
            
            # Load metadata JSON if present
            if 'meta.JSON' in f:
                try:
                    meta_str = f['meta.JSON'][0]
                    if isinstance(meta_str, bytes):
                        meta_str = meta_str.decode('utf-8')
                    exp_data['metadata'] = json.loads(meta_str)
                except Exception as e:
                    logger.warning(f"Could not parse meta.JSON: {e}")
            
            # Compute derived values
            exp_data['n_units'] = len(unit_ids)
            exp_data['total_spikes'] = len(spike_times)
            exp_data['recording_duration'] = float(spike_times.max()) if len(spike_times) > 0 else 1.0
            exp_data['mean_firing_rate'] = np.mean(exp_data['metrics']['firing_rates']) if len(exp_data['metrics'].get('firing_rates', [])) > 0 else 0.0
            exp_data['unit_ids'] = list(unit_ids)
            
    except Exception as e:
        logger.error(f"Error loading experimental data from {h5_path}: {e}")
        traceback.print_exc()
        # Return minimal valid structure
        exp_data = {
            'spike_data': {},
            'metrics': {'firing_rates': np.array([]), 'cv_isi': np.array([])},
            'synchrony': {'mean_pairwise_correlation': 0.0},
            'bursts': [],
            'metadata': {},
            'n_units': 0,
            'total_spikes': 0,
            'recording_duration': 1.0,
            'mean_firing_rate': 0.0,
            'unit_ids': [],
        }
    
    return exp_data


# =============================================================================
# SIMULATED DATA LOADING (from fitnessFunc.py pattern)
# =============================================================================

def get_sim_data_from_call_stack(kwargs: Dict) -> Dict:
    """
    Get the candidate and job path from the call stack.
    This function is used during batch processing to retrieve simulated data.
    
    Parameters
    ----------
    kwargs : dict
        Keyword arguments containing simulation parameters
    
    Returns
    -------
    kwargs : dict
        Updated kwargs with loaded simulation data
    """
    if sim is None:
        raise ImportError("netpyne is required for get_sim_data_from_call_stack")
    
    # get candidate and job path from call stack
    try:
        from .utils.utils_old.extract_simulated_data import get_candidate_and_job_path_from_call_stack
    except ImportError:
        print("Could not import get_candidate_and_job_path_from_call_stack")
        # from RBS_network_models.utils.extract_simulated_data import get_candidate_and_job_path_from_call_stack
    
    candidate_path, candidate_label = get_candidate_and_job_path_from_call_stack()
    fitness_save_path = f'{candidate_path}_fitness.json'
    pkl_path = f'{candidate_path}_data.pkl'
    cfg_path = f'{candidate_path}_cfg.json'
    metrics_path = f'{candidate_path}_metrics.npy'
    
    # load simulated data
    sim.load(pkl_path)
    simData = sim.allSimData.todict().copy()
    popData = sim.net.allPops.copy()
    cellData = sim.net.allCells.copy()
    simCfg = sim.cfg.todict().copy()
    netparams = sim.net.params.todict().copy()
    
    # update kwargs
    kwargs.update({
        'simData': simData,
        'cellData': cellData,
        'popData': popData,
        'simCfg': simCfg,
        'netParams': netparams,
        'simLabel': candidate_label,
        'data_file_path': pkl_path,
        'cfg_file_path': cfg_path,
        'fitness_save_path': fitness_save_path,
        'metrics_save_path': metrics_path,
        'candidate_path': candidate_path,
    })
    
    logger.info(f'Data retrieved from {candidate_path}')
    return kwargs


def get_sim_data_from_simData_pkl(kwargs: Dict) -> Dict:
    """
    Get the candidate and job path from simulated data pkl file.
    
    Parameters
    ----------
    kwargs : dict
        Keyword arguments containing 'sim_data_path' key
    
    Returns
    -------
    kwargs : dict
        Updated kwargs with loaded simulation data
    """
    if sim is None:
        raise ImportError("netpyne is required for get_sim_data_from_simData_pkl")
    
    # load simulated data pkl
    sim_data_path = kwargs.get('sim_data_path', None)
    if sim_data_path is None:
        raise ValueError('No simulated data path found in kwargs.')
    
    sim.load(sim_data_path)
    
    # setup paths
    candidate_path = sim_data_path.replace('_data.pkl', '')
    candidate_label = os.path.basename(candidate_path)
    fitness_save_path = f'{candidate_path}_fitness.json'
    pkl_path = f'{candidate_path}_data.pkl'
    cfg_path = f'{candidate_path}_cfg.json'
    metrics_path = f'{candidate_path}_metrics.npy'
    
    # load simulated data
    simData = sim.allSimData.todict().copy()
    popData = sim.net.allPops.copy()
    cellData = sim.net.allCells.copy()
    simCfg = sim.cfg.todict().copy()
    netparams = sim.net.params.todict().copy()
    
    # update kwargs
    kwargs.update({
        'simData': simData,
        'cellData': cellData,
        'popData': popData,
        'simCfg': simCfg,
        'netParams': netparams,
        'simLabel': candidate_label,
        'data_file_path': pkl_path,
        'cfg_file_path': cfg_path,
        'fitness_save_path': fitness_save_path,
        'metrics_save_path': metrics_path,
        'candidate_path': candidate_path,
    })
    
    logger.info(f'Data retrieved from {candidate_path}')
    return kwargs


def get_simulated_metrics_from_network_analysis(kwargs: Dict) -> Dict:
    """
    Compute network metrics using the same approach as fitnessFunc.py.
    
    Parameters
    ----------
    kwargs : dict
        Keyword arguments containing simulation data
    
    Returns
    -------
    simulated_metrics : dict
        Computed network metrics
    """
    if compute_network_metrics is None:
        logger.warning("compute_network_metrics not available, using basic feature extraction")
        return None
    
    kwargs['source'] = 'simulated'
    simulated_metrics = compute_network_metrics(**kwargs)
    
    # Save metrics if path provided
    metrics_save_path = kwargs.get('metrics_save_path', None)
    if metrics_save_path is not None:
        np.save(metrics_save_path, simulated_metrics)
        logger.info(f'Saved metrics to {metrics_save_path}')
    
    return simulated_metrics


# =============================================================================
# SIMULATED DATA EXTRACTION
# =============================================================================

def extract_simulated_features(simulated_data: Any, recording_duration: Optional[float] = None, min_time: Optional[float] = None) -> Dict:
    """
    Extract features from simulated data (NetPyNE format or dict).
    
    Parameters
    ----------
    simulated_data : Any
        NetPyNE sim object or dict with spike data
    recording_duration : float, optional
        Recording duration in seconds
    min_time : float, optional
        Minimum time to include spikes (seconds). Spikes before this time are excluded.
        Used to exclude network_cool_down period from analysis. Spike times are shifted
        by subtracting min_time so analysis starts at t=0.
    
    Returns
    -------
    features : dict
        Extracted features for fitness computation
    """
    spike_data = {}
    
    try:
        # Handle different input formats
        if isinstance(simulated_data, dict):
            # Check for NetPyNE dict format with 'simData' key
            if 'simData' in simulated_data:
                sim_data = simulated_data['simData']
                spkt = np.array(sim_data.get('spkt', []))
                spkid = np.array(sim_data.get('spkid', []))
                if recording_duration is None:
                    recording_duration = sim_data.get('T', 1000.0) / 1000.0
            else:
                # Direct dict format
                spkt = np.array(simulated_data.get('spkts', simulated_data.get('spkt', [])))
                spkid = np.array(simulated_data.get('spkids', simulated_data.get('spkid', [])))
                if recording_duration is None:
                    recording_duration = simulated_data.get('simDuration', simulated_data.get('T', 1000.0)) / 1000.0
        else:
            # NetPyNE sim object
            if hasattr(simulated_data, 'allSimData'):
                spkt = np.array(simulated_data.allSimData.get('spkt', []))
                spkid = np.array(simulated_data.allSimData.get('spkid', []))
                if recording_duration is None:
                    if hasattr(simulated_data, 'cfg') and hasattr(simulated_data.cfg, 'duration'):
                        recording_duration = simulated_data.cfg.duration / 1000.0
                    else:
                        recording_duration = 1.0
            else:
                logger.warning("Unknown simulated data format")
                spkt = np.array([])
                spkid = np.array([])
                recording_duration = recording_duration or 1.0
        
        # Build spike_data dict
        if len(spkt) > 0 and len(spkid) > 0:
            unique_ids = np.unique(spkid)
            for gid in unique_ids:
                mask = spkid == gid
                # Convert ms to seconds
                spikes_sec = spkt[mask] / 1000.0
                
                # Filter out spikes before min_time (exclude network_cool_down period)
                if min_time is not None and min_time > 0:
                    spikes_sec = spikes_sec[spikes_sec >= min_time]
                    # Shift spike times so analysis starts at t=0
                    spikes_sec = spikes_sec - min_time
                
                spike_data[int(gid)] = spikes_sec
        
        # Compute firing rates
        firing_rates = {}
        for unit_id, spikes in spike_data.items():
            firing_rates[unit_id] = len(spikes) / recording_duration if recording_duration > 0 else 0.0
        
        # Compute CV_ISI
        cv_isi = {}
        for unit_id, spikes in spike_data.items():
            if len(spikes) > 1:
                isis = np.diff(np.sort(spikes))
                if len(isis) > 0 and np.mean(isis) > 0:
                    cv_isi[unit_id] = np.std(isis) / np.mean(isis)
                else:
                    cv_isi[unit_id] = 0.0
            else:
                cv_isi[unit_id] = 0.0
        
        features = {
            'spike_data': spike_data,
            'firing_rates': firing_rates,
            'cv_isi': cv_isi,
            'n_units': len(spike_data),
            'total_spikes': sum(len(spikes) for spikes in spike_data.values()),
            'recording_duration': recording_duration,
            'mean_firing_rate': np.mean(list(firing_rates.values())) if firing_rates else 0.0,
            'unit_ids': list(spike_data.keys()),
        }
        
    except Exception as e:
        logger.error(f"Error extracting simulated features: {e}")
        traceback.print_exc()
        features = {
            'spike_data': {},
            'firing_rates': {},
            'cv_isi': {},
            'n_units': 0,
            'total_spikes': 0,
            'recording_duration': recording_duration or 1.0,
            'mean_firing_rate': 0.0,
            'unit_ids': [],
        }
    
    return features


# =============================================================================
# METRIC COMPUTATION FUNCTIONS
# =============================================================================

def victor_purpura_distance(spikes1: np.ndarray, spikes2: np.ndarray, q: float = 1.0) -> float:
    """
    Compute Victor-Purpura distance between two spike trains.
    
    Parameters
    ----------
    spikes1, spikes2 : np.ndarray
        Spike times in seconds
    q : float
        Temporal precision parameter (1/sec)
    
    Returns
    -------
    distance : float
        VP distance
    """
    n1, n2 = len(spikes1), len(spikes2)
    
    if n1 == 0 and n2 == 0:
        return 0.0
    if n1 == 0:
        return float(n2)
    if n2 == 0:
        return float(n1)
    
    # Dynamic programming
    dp = np.zeros((n1 + 1, n2 + 1))
    dp[:, 0] = np.arange(n1 + 1)
    dp[0, :] = np.arange(n2 + 1)
    
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            shift_cost = q * abs(spikes1[i - 1] - spikes2[j - 1])
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + shift_cost,
            )
    
    return dp[n1, n2]


def compute_spike_timing_score(
    sim_spike_data: Dict[int, np.ndarray],
    exp_spike_data: Dict[int, np.ndarray],
    q_values: List[float] = [0.1, 0.5, 1.0, 2.0],
) -> Dict[str, Any]:
    """
    Compute spike timing metrics between simulated and experimental data.
    
    Parameters
    ----------
    sim_spike_data : dict
        {unit_id: spike_times} for simulated data
    exp_spike_data : dict
        {unit_id: spike_times} for experimental data
    q_values : list
        VP distance q parameters to try
    
    Returns
    -------
    results : dict
        Spike timing metrics
    """
    if len(sim_spike_data) == 0 or len(exp_spike_data) == 0:
        return {
            'mean_distance': np.inf,
            'median_distance': np.inf,
            'std_distance': 0.0,
            'total_score': np.inf,
            'n_comparisons': 0,
        }
    
    # Match units (use minimum set)
    sim_ids = sorted(sim_spike_data.keys())
    exp_ids = sorted(exp_spike_data.keys())
    n_units = min(len(sim_ids), len(exp_ids))
    
    distances = []
    for i in range(n_units):
        sim_spikes = sim_spike_data[sim_ids[i]]
        exp_spikes = exp_spike_data[exp_ids[i]]
        
        # Use best q value (minimum distance)
        min_dist = np.inf
        for q in q_values:
            dist = victor_purpura_distance(sim_spikes, exp_spikes, q)
            if dist < min_dist:
                min_dist = dist
        distances.append(min_dist)
    
    distances = np.array(distances)
    
    return {
        'mean_distance': float(np.mean(distances)),
        'median_distance': float(np.median(distances)),
        'std_distance': float(np.std(distances)),
        'total_score': float(np.mean(distances)),  # Lower is better
        'n_comparisons': n_units,
        'distances': distances.tolist(),
    }


def detect_bursts(
    spike_times: np.ndarray,
    isi_threshold: float = 0.1,
    min_spikes: int = 3,
) -> List[Dict]:
    """
    Detect bursts in a spike train.
    
    Parameters
    ----------
    spike_times : np.ndarray
        Spike times in seconds
    isi_threshold : float
        Maximum ISI within burst
    min_spikes : int
        Minimum spikes for burst
    
    Returns
    -------
    bursts : list
        List of burst dictionaries
    """
    if len(spike_times) < min_spikes:
        return []
    
    spike_times = np.sort(spike_times)
    isis = np.diff(spike_times)
    
    bursts = []
    current_burst = [spike_times[0]]
    
    for i in range(len(isis)):
        if isis[i] <= isi_threshold:
            current_burst.append(spike_times[i + 1])
        else:
            if len(current_burst) >= min_spikes:
                bursts.append({
                    'start': current_burst[0],
                    'end': current_burst[-1],
                    'duration': current_burst[-1] - current_burst[0],
                    'n_spikes': len(current_burst),
                })
            current_burst = [spike_times[i + 1]]
    
    # Check last burst
    if len(current_burst) >= min_spikes:
        bursts.append({
            'start': current_burst[0],
            'end': current_burst[-1],
            'duration': current_burst[-1] - current_burst[0],
            'n_spikes': len(current_burst),
        })
    
    return bursts


def compute_burst_score(
    sim_spike_data: Dict[int, np.ndarray],
    exp_spike_data: Dict[int, np.ndarray],
    exp_bursts: List[Dict],
    recording_duration: float = 1.0,
    isi_threshold: float = 0.1,
    min_spikes: int = 3,
) -> Dict[str, Any]:
    """
    Compute burst metrics comparison.
    
    Parameters
    ----------
    sim_spike_data : dict
        Simulated spike data
    exp_spike_data : dict
        Experimental spike data
    exp_bursts : list
        Pre-computed experimental bursts
    recording_duration : float
        Duration in seconds
    isi_threshold : float
        ISI threshold for burst detection
    min_spikes : int
        Minimum spikes for burst
    
    Returns
    -------
    results : dict
        Burst comparison metrics
    """
    # Detect bursts in simulated data
    sim_all_spikes = np.concatenate(list(sim_spike_data.values())) if sim_spike_data else np.array([])
    sim_all_spikes = np.sort(sim_all_spikes)
    sim_bursts = detect_bursts(sim_all_spikes, isi_threshold, min_spikes)
    
    # Get experimental bursts if not provided
    if not exp_bursts:
        exp_all_spikes = np.concatenate(list(exp_spike_data.values())) if exp_spike_data else np.array([])
        exp_all_spikes = np.sort(exp_all_spikes)
        exp_bursts = detect_bursts(exp_all_spikes, isi_threshold, min_spikes)
    
    # Compute metrics
    n_sim_bursts = len(sim_bursts)
    n_exp_bursts = len(exp_bursts) if isinstance(exp_bursts, list) else 0
    
    sim_burst_rate = n_sim_bursts / recording_duration if recording_duration > 0 else 0
    exp_burst_rate = n_exp_bursts / recording_duration if recording_duration > 0 else 0
    
    # Burst rate error
    rate_error = abs(sim_burst_rate - exp_burst_rate)
    
    # Duration error
    if n_sim_bursts > 0 and n_exp_bursts > 0:
        sim_durations = [b['duration'] for b in sim_bursts]
        exp_durations = [b.get('duration', 0) for b in exp_bursts] if isinstance(exp_bursts, list) else []
        duration_error = abs(np.mean(sim_durations) - np.mean(exp_durations)) if exp_durations else np.inf
    else:
        duration_error = 1.0 if n_sim_bursts != n_exp_bursts else 0.0
    
    # IBI error
    if n_sim_bursts > 1 and n_exp_bursts > 1:
        sim_ibis = [sim_bursts[i+1]['start'] - sim_bursts[i]['end'] for i in range(n_sim_bursts - 1)]
        exp_starts = [b.get('start', 0) for b in exp_bursts]
        exp_ends = [b.get('end', 0) for b in exp_bursts]
        exp_ibis = [exp_starts[i+1] - exp_ends[i] for i in range(n_exp_bursts - 1)] if len(exp_starts) > 1 else []
        ibi_error = abs(np.mean(sim_ibis) - np.mean(exp_ibis)) if exp_ibis else np.inf
    else:
        ibi_error = 1.0
    
    # Combined score (weighted)
    total_score = 0.35 * min(rate_error * 10, 100) + 0.35 * min(duration_error * 100, 100) + 0.30 * min(ibi_error * 10, 100)
    
    return {
        'n_sim_bursts': n_sim_bursts,
        'n_exp_bursts': n_exp_bursts,
        'rate_error': float(rate_error),
        'duration_error': float(duration_error),
        'ibi_error': float(ibi_error),
        'total_score': float(total_score),
        'sim_burst_rate': float(sim_burst_rate),
        'exp_burst_rate': float(exp_burst_rate),
    }


def compute_unit_score(
    sim_features: Dict,
    exp_data: Dict,
    recording_duration: float = 1.0,
) -> Dict[str, Any]:
    """
    Compute unit-level metrics comparison.
    
    Parameters
    ----------
    sim_features : dict
        Simulated features
    exp_data : dict
        Experimental data
    recording_duration : float
        Duration in seconds
    
    Returns
    -------
    results : dict
        Unit-level comparison metrics
    """
    # Get firing rates
    sim_rates = np.array(list(sim_features.get('firing_rates', {}).values()))
    exp_rates = exp_data.get('metrics', {}).get('firing_rates', np.array([]))
    
    if len(sim_rates) == 0 or len(exp_rates) == 0:
        return {
            'firing_rate_error': np.inf,
            'cv_isi_error': np.inf,
            'spike_count_error': np.inf,
            'total_score': np.inf,
        }
    
    # Match lengths
    n = min(len(sim_rates), len(exp_rates))
    sim_rates = np.sort(sim_rates)[::-1][:n]  # Sort descending to match by activity
    exp_rates = np.sort(exp_rates)[::-1][:n]
    
    # Firing rate MAE
    fr_mae = np.mean(np.abs(sim_rates - exp_rates))
    
    # CV_ISI comparison
    sim_cv = np.array(list(sim_features.get('cv_isi', {}).values()))
    exp_cv = exp_data.get('metrics', {}).get('cv_isi', np.array([]))
    
    n_cv = min(len(sim_cv), len(exp_cv))
    if n_cv > 0:
        sim_cv = np.sort(sim_cv)[::-1][:n_cv]
        exp_cv = np.sort(exp_cv)[::-1][:n_cv]
        cv_error = np.mean(np.abs(sim_cv - exp_cv))
    else:
        cv_error = 1.0
    
    # Spike count error
    sim_total = sim_features.get('total_spikes', 0)
    exp_total = exp_data.get('total_spikes', 0)
    if exp_total > 0:
        spike_count_error = abs(sim_total - exp_total) / exp_total
    else:
        spike_count_error = 1.0 if sim_total > 0 else 0.0
    
    # Combined score
    total_score = 0.40 * min(fr_mae, 100) + 0.30 * min(cv_error * 20, 100) + 0.30 * min(spike_count_error * 100, 100)
    
    return {
        'firing_rate_error': float(fr_mae),
        'cv_isi_error': float(cv_error),
        'spike_count_error': float(spike_count_error),
        'total_score': float(total_score),
        'mean_sim_rate': float(np.mean(sim_rates)),
        'mean_exp_rate': float(np.mean(exp_rates)),
        'n_units_compared': n,
    }


def compute_correlation_matrix(
    spike_data: Dict[int, np.ndarray],
    bin_size: float = 0.01,
    recording_duration: float = 1.0,
) -> np.ndarray:
    """
    Compute pairwise correlation matrix for spike trains.
    
    Parameters
    ----------
    spike_data : dict
        {unit_id: spike_times}
    bin_size : float
        Bin size in seconds
    recording_duration : float
        Duration in seconds
    
    Returns
    -------
    corr_matrix : np.ndarray
        Correlation matrix
    """
    unit_ids = sorted(spike_data.keys())
    n_units = len(unit_ids)
    
    if n_units == 0:
        return np.array([[]])
    
    bins = np.arange(0, recording_duration + bin_size, bin_size)
    binned = np.zeros((n_units, len(bins) - 1))
    
    for i, uid in enumerate(unit_ids):
        spikes = spike_data[uid]
        if len(spikes) > 0:
            counts, _ = np.histogram(spikes, bins=bins)
            binned[i, :] = counts
    
    corr_matrix = np.corrcoef(binned)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    return corr_matrix


def detect_adaptive_network_bursts(
    spike_data: Dict[int, np.ndarray],
    gamma: float = 2.0,
    min_burstlet_participation: float = 0.20,
    min_absolute_rate_Hz: float = 0.5,
    min_burst_density_Hz: float = 1.0,
    min_relative_height: float = 0.25,
    bin_ms: float = 10.0,
    smoothing_min_ms: float = 20.0,
    superburst_min_duration_s: float = 2.5,
) -> Dict[str, Any]:
    """
    Detect hierarchical network bursts using adaptive synchrony-based analysis.
    
    This implements the same algorithm as compute_bursting_metrics_v2 in network_analysis.py:
    - Adaptive binning based on biological ISI
    - Synchrony signal (P^gamma) instead of pure firing rate
    - Hierarchical burst structure (burstlets → network_bursts → superbursts)
    - Peak width-based boundary detection
    - Multi-gate filtering system
    
    Parameters
    ----------
    spike_data : dict
        {unit_id: spike_times in seconds}
    gamma : float
        Power for synchrony signal (default: 2.0)
    min_burstlet_participation : float
        Minimum fraction of units participating (default: 0.20)
    min_absolute_rate_Hz : float
        Minimum absolute firing rate (default: 0.5 Hz)
    min_burst_density_Hz : float
        Minimum burst density (default: 1.0 Hz)
    min_relative_height : float
        Minimum peak height relative to global max (default: 0.25)
    bin_ms : float
        Maximum bin size in milliseconds (default: 10.0)
    smoothing_min_ms : float
        Minimum smoothing window in milliseconds (default: 20.0)
    superburst_min_duration_s : float
        Minimum superburst duration in seconds (default: 2.5)
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - burstlets: List of fast, small bursts
        - network_bursts: List of merged medium-scale bursts
        - superbursts: List of long, clustered burst events
        - baseline: Computed from slow smoothed signal
        - diagnostics: Adaptive parameters used
    """
    if len(spike_data) == 0:
        return {
            'burstlets': [], 'network_bursts': [], 'superbursts': [],
            'baseline': 0.0, 'diagnostics': {}
        }
    
    # 0. Sanity checks
    units = list(spike_data.keys())
    all_spikes = np.sort(np.concatenate([spike_data[u] for u in units if len(spike_data[u]) > 0]))
    if all_spikes.size == 0:
        return {
            'burstlets': [], 'network_bursts': [], 'superbursts': [],
            'baseline': 0.0, 'diagnostics': {}
        }
    
    rec_start, rec_end = float(all_spikes[0]), float(all_spikes[-1])
    total_dur = rec_end - rec_start
    
    # 1. ISI-N Calibration (adaptive binning based on biological ISI)
    all_isis = []
    N = 2
    for u in units:
        t = np.unique(np.sort(spike_data[u]))
        if len(t) >= N:
            isin = t[N-1:] - t[:-N+1]
            isin = isin[isin > 0]
            all_isis.extend(isin.tolist())
    
    biological_isi_s = np.median(all_isis) if all_isis else 0.1
    adaptive_bin_ms = np.clip(0.5 * biological_isi_s * 1000.0, 1.0, bin_ms)
    bin_size = adaptive_bin_ms / 1000.0
    bins = np.arange(rec_start, rec_end + bin_size, bin_size)
    t_centers = (bins[:-1] + bins[1:]) / 2.0
    
    # 2. Network Synchrony Signal (number of units active)
    P = np.zeros(len(t_centers))
    for u in units:
        counts, _ = np.histogram(spike_data[u], bins=bins)
        P += (counts > 0).astype(float)
    W = P ** gamma
    
    # 3. Dual Smoothing (fast for onset, slow for baseline)
    sigma_min_bins = max(1.0, smoothing_min_ms / adaptive_bin_ms)
    sigma_slow_bins = max(sigma_min_bins, 5.0 * biological_isi_s / bin_size)
    sigma_fast_bins = np.clip(sigma_slow_bins / 5.0, 1.0, 5.0)
    
    ws_onset = gaussian_filter1d(W, sigma_fast_bins)
    ws_peak = gaussian_filter1d(W, sigma_slow_bins)
    
    # Adaptive merge gaps
    burstlet_merge_gap_s = 2.0 * biological_isi_s
    network_merge_gap_s = 5.0 * biological_isi_s
    
    # Baseline from slow signal
    baseline = np.median(ws_peak)
    
    # 4. Peak-based burst detection
    global_max_height = np.max(ws_onset) if len(ws_onset) > 0 else 0
    relative_threshold_val = global_max_height * min_relative_height
    min_peak_height = np.max(ws_onset) * 0.01
    
    peaks, properties = find_peaks(ws_onset, height=min_peak_height)
    
    if len(peaks) == 0:
        return {
            'burstlets': [], 'network_bursts': [], 'superbursts': [],
            'baseline': baseline, 
            'diagnostics': {
                'adaptive_bin_ms': adaptive_bin_ms,
                'biological_isi_s': biological_isi_s,
                'total_duration_s': total_dur,
                'num_units': len(units)
            }
        }
    
    # Get burst boundaries from peak widths (at 80% height)
    results_width = peak_widths(ws_onset, peaks, rel_height=0.80)
    starts_idx = np.clip(np.floor(results_width[2]).astype(int), 0, len(ws_onset)-1)
    ends_idx = np.clip(np.ceil(results_width[3]).astype(int), 0, len(ws_onset)-1)
    
    # 5. Filter and collect burstlets
    burstlets = []
    for i in range(len(peaks)):
        s, e = starts_idx[i], ends_idx[i]
        if s == e:
            e += 1
        
        start_t = t_centers[s]
        end_t = t_centers[min(e, len(t_centers)-1)]
        duration_s = end_t - start_t
        
        if duration_s <= 0:
            continue
        
        # Calculate metrics
        participating = 0
        total_spikes = 0
        for u in units:
            ts = spike_data[u]
            c = np.sum((ts >= start_t) & (ts <= end_t))
            total_spikes += c
            if c > 0:
                participating += 1
        
        participation_frac = participating / len(units)
        denom = (duration_s * max(1, participating))
        burst_density = total_spikes / denom if denom > 0 else 0
        peak_fast = np.max(ws_onset[s:e]) if (e > s) else 0
        
        # Multi-gate filtering
        if peak_fast < relative_threshold_val:
            continue
        if participation_frac < min_burstlet_participation:
            continue
        if burst_density < min_burst_density_Hz:
            continue
        
        burstlets.append({
            'start': float(start_t),
            'end': float(end_t),
            'duration_s': float(duration_s),
            'peak_synchrony': float(peak_fast),
            'synchrony_energy': float(np.sum(ws_peak[s:e]) * bin_size),
            'participation': participation_frac,
            'total_spikes': int(total_spikes)
        })
    
    # 6. Hierarchical merging
    def merge_bursts(events, gap, min_dur=0):
        if not events:
            return []
        events = sorted(events, key=lambda x: x['start'])
        
        merged = []
        curr = [events[0]]
        s, e = events[0]['start'], events[0]['end']
        
        for nxt in events[1:]:
            if nxt['start'] <= e + gap:
                curr.append(nxt)
                e = max(e, nxt['end'])
            else:
                if (e - s) >= min_dur:
                    merged.append({
                        'start': s, 'end': e, 'duration_s': e - s,
                        'peak_synchrony': max(ev['peak_synchrony'] for ev in curr),
                        'synchrony_energy': sum(ev['synchrony_energy'] for ev in curr),
                        'fragment_count': len(curr),
                        'total_spikes': sum(ev['total_spikes'] for ev in curr),
                        'participation': float(np.mean([ev['participation'] for ev in curr]))
                    })
                curr = [nxt]
                s, e = nxt['start'], nxt['end']
        
        if (e - s) >= min_dur:
            merged.append({
                'start': s, 'end': e, 'duration_s': e - s,
                'peak_synchrony': max(ev['peak_synchrony'] for ev in curr),
                'synchrony_energy': sum(ev['synchrony_energy'] for ev in curr),
                'fragment_count': len(curr),
                'total_spikes': sum(ev['total_spikes'] for ev in curr),
                'participation': float(np.mean([ev['participation'] for ev in curr]))
            })
        
        return merged
    
    network_bursts = merge_bursts(burstlets, burstlet_merge_gap_s)
    superbursts = merge_bursts(network_bursts, network_merge_gap_s, superburst_min_duration_s)
    
    return {
        'burstlets': burstlets,
        'network_bursts': network_bursts,
        'superbursts': superbursts,
        'baseline': baseline,
        'diagnostics': {
            'adaptive_bin_ms': adaptive_bin_ms,
            'biological_isi_s': biological_isi_s,
            'total_duration_s': total_dur,
            'num_units': len(units),
            'burstlet_merge_gap_s': burstlet_merge_gap_s,
            'network_merge_gap_s': network_merge_gap_s
        }
    }


def compute_hierarchical_burst_level_score(
    sim_bursts: List[Dict],
    exp_bursts_data: Any,
    recording_duration: float,
    level_name: str = 'network_bursts',
) -> Dict[str, Any]:
    """
    Compute score for a single hierarchical burst level (burstlets, network_bursts, or superbursts).
    
    Parameters
    ----------
    sim_bursts : list
        List of simulated burst dictionaries from detect_adaptive_network_bursts
    exp_bursts_data : dict or list
        Experimental burst data (can be dict with metrics or list of burst events)
    recording_duration : float
        Recording duration in seconds
    level_name : str
        Name of the hierarchical level ('burstlets', 'network_bursts', 'superbursts')
    
    Returns
    -------
    results : dict
        Burst comparison metrics for this hierarchical level
    """
    # Compute simulated metrics
    sim_num_bursts = len(sim_bursts)
    sim_burst_rate = sim_num_bursts / recording_duration if recording_duration > 0 else 0
    
    # Extract experimental metrics
    if isinstance(exp_bursts_data, dict):
        # From bursting_data_v2 structure
        exp_num_bursts = exp_bursts_data.get('num_bursts', 0)
        exp_burst_rate = exp_bursts_data.get('burst_rate', 0.0)
    elif isinstance(exp_bursts_data, list):
        # Direct list of bursts
        exp_num_bursts = len(exp_bursts_data)
        exp_burst_rate = exp_num_bursts / recording_duration if recording_duration > 0 else 0
    else:
        # No data
        exp_num_bursts = 0
        exp_burst_rate = 0.0
    
    # === Rate Error ===
    rate_error = abs(sim_burst_rate - exp_burst_rate)
    
    # === Count Error ===
    count_error = abs(sim_num_bursts - exp_num_bursts) / max(exp_num_bursts, 1)
    
    # === Duration Error ===
    if sim_num_bursts > 0 and exp_num_bursts > 0:
        sim_durations = [b['duration_s'] for b in sim_bursts]
        
        if isinstance(exp_bursts_data, dict):
            exp_dur_stats = exp_bursts_data.get('burst_duration', {})
            if isinstance(exp_dur_stats, dict) and 'mean' in exp_dur_stats:
                exp_mean_duration = exp_dur_stats['mean']
                mean_duration_error = abs(np.mean(sim_durations) - exp_mean_duration)
            else:
                mean_duration_error = 1.0
        elif isinstance(exp_bursts_data, list):
            if isinstance(exp_bursts_data[0], dict):
                exp_durations = [b.get('duration_s', b.get('end', 0) - b.get('start', 0)) 
                               for b in exp_bursts_data]
            else:
                exp_durations = [end - start for start, end in exp_bursts_data]
            mean_duration_error = abs(np.mean(sim_durations) - np.mean(exp_durations))
        else:
            mean_duration_error = 1.0
    else:
        mean_duration_error = 1.0 if sim_num_bursts != exp_num_bursts else 0.0
    
    # === IBI Error (only for network_bursts and superbursts) ===
    if level_name in ['network_bursts', 'superbursts']:
        if sim_num_bursts > 1 and exp_num_bursts > 1:
            sim_starts = [b['start'] for b in sim_bursts]
            sim_ibis = np.diff(sim_starts)
            
            if isinstance(exp_bursts_data, dict):
                exp_ibi_stats = exp_bursts_data.get('ibi', {})
                if isinstance(exp_ibi_stats, dict) and 'mean' in exp_ibi_stats:
                    exp_mean_ibi = exp_ibi_stats['mean']
                    ibi_error = abs(np.mean(sim_ibis) - exp_mean_ibi)
                else:
                    ibi_error = 1.0
            elif isinstance(exp_bursts_data, list) and len(exp_bursts_data) > 1:
                if isinstance(exp_bursts_data[0], dict):
                    exp_starts = [b.get('start', 0) for b in exp_bursts_data]
                else:
                    exp_starts = [start for start, end in exp_bursts_data]
                exp_ibis = np.diff(exp_starts)
                ibi_error = abs(np.mean(sim_ibis) - np.mean(exp_ibis))
            else:
                ibi_error = 1.0
        else:
            ibi_error = 1.0
    else:
        ibi_error = 0.0  # Not applicable for burstlets
    
    # === Participation Error (only for network_bursts) ===
    if level_name == 'network_bursts' and sim_num_bursts > 0:
        sim_mean_participation = np.mean([b['participation'] for b in sim_bursts])
        
        if isinstance(exp_bursts_data, dict):
            exp_part_stats = exp_bursts_data.get('participation', {})
            if isinstance(exp_part_stats, dict) and 'mean' in exp_part_stats:
                exp_mean_participation = exp_part_stats['mean']
                participation_error = abs(sim_mean_participation - exp_mean_participation)
            else:
                participation_error = 0.5
        elif isinstance(exp_bursts_data, list) and len(exp_bursts_data) > 0:
            if isinstance(exp_bursts_data[0], dict) and 'participation' in exp_bursts_data[0]:
                exp_mean_participation = np.mean([b.get('participation', 0.5) for b in exp_bursts_data])
                participation_error = abs(sim_mean_participation - exp_mean_participation)
            else:
                participation_error = 0.5
        else:
            participation_error = 0.5
    else:
        participation_error = 0.0  # Not applicable
    
    # === Synchrony Energy Error (only for network_bursts) ===
    if level_name == 'network_bursts' and sim_num_bursts > 0:
        sim_mean_energy = np.mean([b['synchrony_energy'] for b in sim_bursts])
        
        if isinstance(exp_bursts_data, dict):
            exp_energy_stats = exp_bursts_data.get('synchrony_energy', {})
            if isinstance(exp_energy_stats, dict) and 'mean' in exp_energy_stats:
                exp_mean_energy = exp_energy_stats['mean']
                if exp_mean_energy > 0:
                    energy_error = abs(sim_mean_energy - exp_mean_energy) / exp_mean_energy
                else:
                    energy_error = 0.5
            else:
                energy_error = 0.5
        elif isinstance(exp_bursts_data, list) and len(exp_bursts_data) > 0:
            if isinstance(exp_bursts_data[0], dict) and 'synchrony_energy' in exp_bursts_data[0]:
                exp_mean_energy = np.mean([b.get('synchrony_energy', 1.0) for b in exp_bursts_data])
                energy_error = abs(sim_mean_energy - exp_mean_energy) / max(exp_mean_energy, 0.001)
            else:
                energy_error = 0.5
        else:
            energy_error = 0.5
    else:
        energy_error = 0.0  # Not applicable
    
    # === Compute total score based on level ===
    if level_name == 'burstlets':
        total_score = (
            0.50 * min(rate_error * 10, 100) +
            0.30 * min(count_error * 100, 100) +
            0.20 * min(mean_duration_error * 100, 100)
        )
    elif level_name == 'network_bursts':
        total_score = (
            0.25 * min(rate_error * 10, 100) +
            0.20 * min(count_error * 100, 100) +
            0.20 * min(mean_duration_error * 100, 100) +
            0.15 * min(ibi_error * 10, 100) +
            0.10 * min(participation_error * 100, 100) +
            0.10 * min(energy_error * 100, 100)
        )
    elif level_name == 'superbursts':
        total_score = (
            0.40 * min(rate_error * 5, 100) +
            0.30 * min(count_error * 100, 100) +
            0.30 * min(mean_duration_error * 50, 100)
        )
    else:
        total_score = 100.0  # Unknown level
    
    return {
        'sim_num_bursts': sim_num_bursts,
        'exp_num_bursts': exp_num_bursts,
        'sim_burst_rate': float(sim_burst_rate),
        'exp_burst_rate': float(exp_burst_rate),
        'rate_error': float(rate_error),
        'count_error': float(count_error),
        'duration_error': float(mean_duration_error),
        'ibi_error': float(ibi_error),
        'participation_error': float(participation_error),
        'energy_error': float(energy_error),
        'total_score': float(total_score),
    }


def compute_network_burst_score(
    sim_spike_data: Dict[int, np.ndarray],
    exp_data: Dict,
    recording_duration: float = 1.0,
    use_hierarchical: bool = True,
    bin_size: float = None,  # Legacy parameter (converted to bin_ms)
    threshold_std: float = None,  # Legacy parameter (ignored in adaptive method)
    **burst_params
) -> Dict[str, Any]:
    """
    Compute hierarchical burst metrics for all three levels: burstlets, network_bursts, superbursts.
    
    Parameters
    ----------
    sim_spike_data : dict
        Simulated spike data
    exp_data : dict
        Experimental data
    recording_duration : float
        Duration in seconds
    use_hierarchical : bool
        If True, use adaptive hierarchical detection (default)
    bin_size : float, optional
        Legacy parameter - converted to bin_ms (bin_size * 1000)
    threshold_std : float, optional
        Legacy parameter - ignored in adaptive detection
    **burst_params : dict
        Additional parameters for burst detection (gamma, min_burstlet_participation, etc.)
    
    Returns
    -------
    results : dict
        Hierarchical burst comparison metrics for all three levels
    """
    # Convert legacy bin_size parameter to bin_ms if provided
    if bin_size is not None and 'bin_ms' not in burst_params:
        burst_params['bin_ms'] = bin_size * 1000.0  # Convert seconds to milliseconds
    
    # Remove any other legacy parameters that don't match the new function signature
    legacy_params = ['threshold_std', 'min_duration_bins']
    for param in legacy_params:
        burst_params.pop(param, None)
    
    # Detect hierarchical bursts in simulated data
    sim_burst_results = detect_adaptive_network_bursts(sim_spike_data, **burst_params)
    
    # Extract all three hierarchical levels
    sim_burstlets = sim_burst_results['burstlets']
    sim_network_bursts = sim_burst_results['network_bursts']
    sim_superbursts = sim_burst_results['superbursts']
    
    # Get experimental hierarchical burst data
    exp_bursting_v2 = exp_data.get('bursting_data_v2', {})
    
    # Extract experimental data for each level
    if exp_bursting_v2:
        exp_burstlets = exp_bursting_v2.get('burstlets', {})
        exp_network_bursts = exp_bursting_v2.get('network_bursts', {})
        exp_superbursts = exp_bursting_v2.get('superbursts', {})
    else:
        # Fall back to synchrony data for network bursts only
        exp_sync = exp_data.get('synchrony', {})
        exp_burstlets = {}
        exp_network_bursts = exp_sync.get('network_bursts', [])
        exp_superbursts = {}
    
    # Compute scores for each hierarchical level
    burstlet_score = compute_hierarchical_burst_level_score(
        sim_burstlets, exp_burstlets, recording_duration, 'burstlets'
    )
    
    network_burst_score = compute_hierarchical_burst_level_score(
        sim_network_bursts, exp_network_bursts, recording_duration, 'network_bursts'
    )
    
    superburst_score = compute_hierarchical_burst_level_score(
        sim_superbursts, exp_superbursts, recording_duration, 'superbursts'
    )
    
    # Return combined results
    return {
        'burstlets': burstlet_score,
        'network_bursts': network_burst_score,
        'superbursts': superburst_score,
        'diagnostics': sim_burst_results['diagnostics'],
        'baseline': sim_burst_results['baseline'],
        # Legacy compatibility - use network_bursts score as total
        'total_score': network_burst_score['total_score'],
    }


def compute_synchrony_score(
    sim_spike_data: Dict[int, np.ndarray],
    exp_data: Dict,
    bin_size: float = 0.01,
    recording_duration: float = 1.0,
) -> Dict[str, Any]:
    """
    Compute network synchrony metrics.
    
    Parameters
    ----------
    sim_spike_data : dict
        Simulated spike data
    exp_data : dict
        Experimental data
    bin_size : float
        Bin size in seconds
    recording_duration : float
        Duration in seconds
    
    Returns
    -------
    results : dict
        Synchrony comparison metrics
    """
    # Compute simulated correlation matrix
    sim_corr = compute_correlation_matrix(sim_spike_data, bin_size, recording_duration)
    
    # Get experimental synchrony
    exp_sync = exp_data.get('synchrony', {})
    exp_mean_corr = exp_sync.get('mean_pairwise_correlation', 0.0)
    
    # Compute simulated mean pairwise correlation (excluding diagonal)
    if sim_corr.size > 1:
        n = sim_corr.shape[0]
        triu_idx = np.triu_indices(n, k=1)
        sim_mean_corr = np.mean(sim_corr[triu_idx])
    else:
        sim_mean_corr = 0.0
    
    # Correlation error
    corr_error = abs(sim_mean_corr - exp_mean_corr)
    
    # Global synchrony (variance of population rate)
    if sim_spike_data:
        bins = np.arange(0, recording_duration + bin_size, bin_size)
        pop_rate = np.zeros(len(bins) - 1)
        for spikes in sim_spike_data.values():
            counts, _ = np.histogram(spikes, bins=bins)
            pop_rate += counts
        sim_sync_index = np.var(pop_rate) / np.mean(pop_rate) if np.mean(pop_rate) > 0 else 0
    else:
        sim_sync_index = 0
    
    exp_sync_index = exp_sync.get('synchrony_index', 0)
    sync_error = abs(sim_sync_index - exp_sync_index) / max(exp_sync_index, 1)
    
    # Combined score
    total_score = 0.50 * min(corr_error * 100, 100) + 0.50 * min(sync_error * 100, 100)
    
    return {
        'correlation_matrix_error': float(corr_error),
        'global_synchrony_error': float(sync_error),
        'total_score': float(total_score),
        'sim_mean_correlation': float(sim_mean_corr),
        'exp_mean_correlation': float(exp_mean_corr),
        'sim_synchrony_index': float(sim_sync_index),
        'exp_synchrony_index': float(exp_sync_index),
    }


# =============================================================================
# MAIN FITNESS FUNCTION
# =============================================================================

def fitnessFunc_claude(
    simulated_data_path: str = None,
    reference_data_path: str = None,
    weights: Optional[Dict[str, float]] = None,
    config: Optional[Dict] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main fitness function for network optimization.
    
    Computes multi-metric fitness comparing simulated data to experimental reference.
    Returns a dictionary compatible with the fitnessFunc_v3 structure for batch optimization.
    
    Parameters
    ----------
    simulated_data_path : str, optional
        Path to simulated data file (e.g., _data.pkl or .npy). Can also be None if batching=True.
    reference_data_path : str
        Path to experimental_data.h5 file
    weights : dict, optional
        Component weights (default: spike_timing=0.30, burst=0.25, unit=0.20, 
        synchrony=0.10, network_burst=0.15)
    config : dict, optional
        Configuration for metric computation
    **kwargs : dict
        Additional arguments:
        - batching: bool - If True, loads data from call stack (batch processing mode)
        - sim_data_path: str - Alternative path to simulated data pkl file
        - use_network_metrics: bool - If True, uses compute_network_metrics for detailed analysis
    
    Returns
    -------
    result : dict
        Fitness result dictionary with structure:
        {
            'fitness': float,  # Overall fitness score (lower is better)
            'fitness_dict': {...},  # Detailed breakdown
            'simulated_metrics': {...},  # Simulated network metrics
            'experimental_metrics': {...},  # Experimental reference metrics
        }
    """
    time_start = time.time()
    
    # Default weights
    if weights is None:
        try:
            weights = get_component_weights()
        except:
            # Fallback default weights if schema not available
            weights = {
                'spike_timing': 0.30,
                'burst_metrics': 0.25,
                'unit_metrics': 0.20,
                'synchrony': 0.10,
                'network_burst': 0.15,
            }
    
    # Normalize weights to sum to 1.0
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    # Default configuration
    if config is None:
        config = DEFAULT_CONFIG.copy()
    simulated_data = None
    try:
        # Set source to simulated
        kwargs['source'] = 'simulated'
        
        # Handle batching mode - load simulated data from call stack or pkl file
        if kwargs.get('batching', False):
            # logger.info("Batching mode enabled - loading data from call stack")
            kwargs = get_sim_data_from_call_stack(kwargs)
            simulated_data = simulated_data_path
            if type(simulated_data_path) == str:
                simulated_data_path = kwargs.get('data_file_path', simulated_data_path)
                simulated_data = None
            simulated_data = kwargs.get('simData', None)
        elif kwargs.get('sim_data_path', None) is not None:
            logger.info(f"Loading simulated data from pkl file: {kwargs['sim_data_path']}")
            kwargs = get_sim_data_from_simData_pkl(kwargs)
            simulated_data_path = kwargs.get('data_file_path', kwargs.get('sim_data_path'))
            simulated_data = kwargs.get('simData', None)
        else:
            # Load simulated data from file path
            simulated_data = None
            if simulated_data_path is not None:
                if isinstance(simulated_data_path, str):
                    if simulated_data_path.endswith('.pkl'):
                        # Load pickle file using netpyne
                        if sim is not None:
                            sim.load(simulated_data_path)
                            simulated_data = sim.allSimData.todict().copy()
                            kwargs['simData'] = simulated_data
                            kwargs['popData'] = sim.net.allPops.copy()
                            kwargs['cellData'] = sim.net.allCells.copy()
                            kwargs['simCfg'] = sim.cfg.todict().copy()
                            kwargs['netParams'] = sim.net.params.todict().copy()
                        else:
                            logger.warning("netpyne not available, trying numpy load")
                            simulated_data = np.load(simulated_data_path, allow_pickle=True)
                    else:
                        simulated_data = np.load(simulated_data_path, allow_pickle=True)
                elif isinstance(simulated_data_path, dict):
                    simulated_data = simulated_data_path
        
        # Compute network metrics if available and requested
        use_network_metrics = kwargs.get('use_network_metrics', False)
        simulated_metrics = None
        if use_network_metrics and compute_network_metrics is not None and kwargs.get('simData') is not None:
            logger.info("Computing network metrics using compute_network_metrics")
            simulated_metrics = get_simulated_metrics_from_network_analysis(kwargs)
        
        # Load experimental data
        logger.info(f"Loading experimental data from {reference_data_path}")
        exp_data = load_experimental_h5(reference_data_path)
        
        if exp_data['n_units'] == 0:
            logger.warning("No units found in experimental data")
            # return _create_error_result("No experimental units found", kwargs)
            return 1000.0  # Return high fitness score directly
        
        # Extract simulated features
        logger.info("Extracting simulated features")
        recording_duration = exp_data['recording_duration']/1000.0  # convert ms to s
        
        # Get network_cool_down parameter - this is the initial stabilization period to exclude
        network_cool_down = kwargs.get('network_cool_down', 0.0)
        if network_cool_down > 0:
            logger.info(f"Network cool down period: {network_cool_down}s - excluding initial stabilization from analysis")
        min_analysis_time = network_cool_down  # Exclude spikes before this time
        
        # Use simulated_data that was already loaded above (from batching, pkl, or direct load)
        if simulated_data is None:
            # If still None, try loading from path as fallback
            if simulated_data_path is not None and isinstance(simulated_data_path, str):
                if simulated_data_path.endswith('.pkl'):
                    if sim is not None:
                        logger.info(f"Loading simulated data from pkl file after None: {simulated_data_path}")
                        sim.load(simulated_data_path)
                        simulated_data = sim.allSimData.todict().copy()
                    else:
                        simulated_data = np.load(simulated_data_path, allow_pickle=True)
                else:
                    simulated_data = np.load(simulated_data_path, allow_pickle=True)
            elif isinstance(simulated_data_path, dict):
                simulated_data = simulated_data_path
        
        if simulated_data is None:
            logger.warning("No simulated data provided")
            return 1000.0  # Return high fitness score directly
        
        sim_features = extract_simulated_features(simulated_data, recording_duration, min_time=min_analysis_time)
        
        if sim_features['n_units'] == 0:
            logger.warning("No spikes in simulation")
            # return _create_error_result("No spikes in simulation", kwargs)
            return 1000.0  # Return high fitness score directly
        
        # Compute metrics
        logger.info("Computing spike timing metrics")
        spike_timing_results = compute_spike_timing_score(
            sim_features['spike_data'],
            exp_data['spike_data'],
            q_values=config.get('spike_timing', {}).get('q_values', [0.1, 0.5, 1.0, 2.0]),
        )
        
        logger.info("Computing burst metrics")
        burst_results = compute_burst_score(
            sim_features['spike_data'],
            exp_data['spike_data'],
            exp_data.get('bursts', []),
            recording_duration=recording_duration,
            isi_threshold=config.get('burst', {}).get('isi_threshold', 0.1),
            min_spikes=config.get('burst', {}).get('min_spikes', 3),
        )
        
        logger.info("Computing unit metrics")
        unit_results = compute_unit_score(
            sim_features,
            exp_data,
            recording_duration=recording_duration,
        )
        
        logger.info("Computing synchrony metrics")
        synchrony_results = compute_synchrony_score(
            sim_features['spike_data'],
            exp_data,
            bin_size=config.get('synchrony', {}).get('bin_size', 0.01),
            recording_duration=recording_duration,
        )
        
        logger.info("Computing hierarchical burst metrics (burstlets, network_bursts, superbursts)")
        hierarchical_burst_results = compute_network_burst_score(
            sim_features['spike_data'],
            exp_data,
            recording_duration=recording_duration,
            **config.get('hierarchical_bursts', {}),
        )
        
        # Extract scores for each hierarchical level
        burstlet_score = hierarchical_burst_results['burstlets']['total_score']
        network_burst_score = hierarchical_burst_results['network_bursts']['total_score']
        superburst_score = hierarchical_burst_results['superbursts']['total_score']
        
        # Compute weighted fitness
        spike_timing_score = spike_timing_results['total_score']
        burst_score = burst_results['total_score']
        unit_score = unit_results['total_score']
        synchrony_score = synchrony_results['total_score']
        
        # Handle infinite values
        max_score = config.get('max_fitness', 1000.0)
        spike_timing_score = min(spike_timing_score, max_score) if np.isfinite(spike_timing_score) else max_score
        burst_score = min(burst_score, max_score) if np.isfinite(burst_score) else max_score
        unit_score = min(unit_score, max_score) if np.isfinite(unit_score) else max_score
        synchrony_score = min(synchrony_score, max_score) if np.isfinite(synchrony_score) else max_score
        burstlet_score = min(burstlet_score, max_score) if np.isfinite(burstlet_score) else max_score
        network_burst_score = min(network_burst_score, max_score) if np.isfinite(network_burst_score) else max_score
        superburst_score = min(superburst_score, max_score) if np.isfinite(superburst_score) else max_score
        
        # Weighted combination with hierarchical burst components
        fitness = (
            weights.get('spike_timing', 0.0) * spike_timing_score +
            weights.get('burst_metrics', 0.05) * burst_score +
            weights.get('unit_metrics', 0.30) * unit_score +
            weights.get('synchrony', 0.10) * synchrony_score +
            weights.get('burstlets', 0.10) * burstlet_score +
            weights.get('network_bursts', 0.35) * network_burst_score +
            weights.get('superbursts', 0.10) * superburst_score
        )
        
        # Cap at max fitness
        fitness = min(fitness, max_score)
        
        # Build result dictionary (compatible with fitnessFunc_v3)
        result = {
            'fitness': float(fitness),
            'fit': float(fitness),  # Legacy compatibility
            'fitness_dict': {
                'spike_timing': {
                    'fit': float(spike_timing_score),
                    **spike_timing_results,
                },
                'burst_metrics': {
                    'fit': float(burst_score),
                    **burst_results,
                },
                'unit_metrics': {
                    'fit': float(unit_score),
                    **unit_results,
                },
                'synchrony': {
                    'fit': float(synchrony_score),
                    **synchrony_results,
                },
                'burstlets': {
                    'fit': float(burstlet_score),
                    **hierarchical_burst_results['burstlets'],
                },
                'network_bursts': {
                    'fit': float(network_burst_score),
                    **hierarchical_burst_results['network_bursts'],
                },
                'superbursts': {
                    'fit': float(superburst_score),
                    **hierarchical_burst_results['superbursts'],
                },
                'hierarchical_diagnostics': hierarchical_burst_results['diagnostics'],
                'fit': float(fitness),
            },
            'simulated_metrics': {
                'n_units': sim_features['n_units'],
                'total_spikes': sim_features['total_spikes'],
                'mean_firing_rate': sim_features['mean_firing_rate'],
                'recording_duration': recording_duration,
                # Include detailed network metrics if computed
                **(simulated_metrics if simulated_metrics is not None else {}),
            },
            'experimental_metrics': {
                'n_units': exp_data['n_units'],
                'total_spikes': exp_data['total_spikes'],
                'mean_firing_rate': exp_data['mean_firing_rate'],
                'recording_duration': exp_data['recording_duration'],
            },
            'weights': weights,
            'config': config,
        }
        
        logger.info(f"Fitness computed: {fitness:.4f}")
        sim_data_path_kwargs = kwargs.get('sim_data_path', None)
        if sim_data_path_kwargs is None:
            logger.info(f"Simulated data path from kwargs: {sim_data_path_kwargs}")
        
        # Get candidate path from kwargs (set by batching functions) or from simulated_data_path
        candidate_path = kwargs.get('candidate_path', None)
        print("SIMULATED DATA PATH:", sim_data_path_kwargs)
        # simulated_data_path = simulated_data_path
        if candidate_path is None and sim_data_path_kwargs is not None and isinstance(sim_data_path_kwargs, str):
            candidate_path = sim_data_path_kwargs.replace('_data.pkl', '').replace('.npy', '')
        
        # Get fitness save path from kwargs or construct from candidate path
        fitness_save_path = kwargs.get('fitness_save_path', None)
        if fitness_save_path is None and candidate_path is not None:
            fitness_save_path = f'{candidate_path}_fitness.json'
        
        # Save fitness to .json file
        if fitness_save_path is not None:
            with open(fitness_save_path, 'w') as f:
                json.dump(result, f, indent=4)
            logger.info(f'Saved fitness to {fitness_save_path}')
        
        # Save network metrics if path provided
        metrics_save_path = kwargs.get('metrics_save_path', None)
        if metrics_save_path is None and candidate_path is not None:
            metrics_save_path = f'{candidate_path}_metrics.npy'
        if metrics_save_path is not None and simulated_metrics is not None:
            np.save(metrics_save_path, simulated_metrics)
            logger.info(f'Saved metrics to {metrics_save_path}')
        
        elapsed_time = time.time() - time_start
        logger.info(f'Elapsed time: {elapsed_time:.2f} seconds.')
        fitness_value = float(result.get('fit', 1000.0))
        print(f"Fitness value: {fitness_value}")
        return fitness_value
        
    except Exception as e:
        logger.error(f"Error computing fitness: {e}")
        traceback.print_exc()
        # return _create_error_result(str(e), kwargs)
        return 1000.0  # Return high fitness score directly


def _create_error_result(error_msg: str, kwargs: Dict) -> Dict[str, Any]:
    """Create error result with max fitness."""
    max_fitness = kwargs.get('max_fitness', 1000.0)
    return {
        'fitness': max_fitness,
        'fit': max_fitness,
        'fitness_dict': {
            'spike_timing': {'fit': max_fitness},
            'burst_metrics': {'fit': max_fitness},
            'unit_metrics': {'fit': max_fitness},
            'synchrony': {'fit': max_fitness},
            'burstlets': {'fit': max_fitness},
            'network_bursts': {'fit': max_fitness},
            'superbursts': {'fit': max_fitness},
            'fit': max_fitness,
            'error': error_msg,
        },
        'simulated_metrics': {},
        'experimental_metrics': {},
        'error': error_msg,
    }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compute_fitness(
    sim_data: Any,
    exp_h5_path: str,
    weights: Optional[Dict] = None,
) -> Tuple[float, Dict]:
    """
    Convenience function to compute fitness.
    
    Parameters
    ----------
    sim_data : Any
        Simulated data
    exp_h5_path : str
        Path to experimental HDF5 file
    weights : dict, optional
        Component weights
    
    Returns
    -------
    fitness : float
        Overall fitness score
    metrics : dict
        Detailed breakdown
    """
    result = fitnessFunc_claude(sim_data, exp_h5_path, weights=weights)
    return result['fitness'], result


def fitness_report(result: Dict) -> str:
    """
    Generate human-readable fitness report.
    
    Parameters
    ----------
    result : dict
        Result from fitnessFunc_claude
    
    Returns
    -------
    report : str
        Formatted report
    """
    lines = []
    lines.append("=" * 70)
    lines.append("CLAUDE FITNESS FUNCTION REPORT")
    lines.append("=" * 70)
    
    fitness = result.get('fitness', 'N/A')
    lines.append(f"\nOVERALL FITNESS: {fitness:.6f}" if isinstance(fitness, float) else f"\nOVERALL FITNESS: {fitness}")
    
    lines.append("\n" + "-" * 70)
    lines.append("COMPONENT SCORES:")
    lines.append("-" * 70)
    
    fitness_dict = result.get('fitness_dict', {})
    weights = result.get('weights', {})
    
    components = [
        ('Spike Timing', 'spike_timing', 'spike_timing'),
        ('Burst Metrics', 'burst_metrics', 'burst_metrics'),
        ('Unit Metrics', 'unit_metrics', 'unit_metrics'),
        ('Synchrony', 'synchrony', 'synchrony'),
        ('Burstlets', 'burstlets', 'burstlets'),
        ('Network Bursts', 'network_bursts', 'network_bursts'),
        ('Superbursts', 'superbursts', 'superbursts'),
    ]
    
    for name, key, weight_key in components:
        score = fitness_dict.get(key, {}).get('fit', 0.0)
        weight = weights.get(weight_key, 0.0)
        contribution = score * weight
        lines.append(f"  {name:20s}: {score:10.4f} (weight={weight:.2f}, contribution={contribution:.4f})")
    
    lines.append("\n" + "-" * 70)
    lines.append("DATA SUMMARY:")
    lines.append("-" * 70)
    
    sim_metrics = result.get('simulated_metrics', {})
    exp_metrics = result.get('experimental_metrics', {})
    
    lines.append(f"  Simulation units: {sim_metrics.get('n_units', 0)}")
    lines.append(f"  Experimental units: {exp_metrics.get('n_units', 0)}")
    lines.append(f"  Simulation spikes: {sim_metrics.get('total_spikes', 0)}")
    lines.append(f"  Experimental spikes: {exp_metrics.get('total_spikes', 0)}")
    lines.append(f"  Simulation mean FR: {sim_metrics.get('mean_firing_rate', 0):.2f} Hz")
    lines.append(f"  Experimental mean FR: {exp_metrics.get('mean_firing_rate', 0):.2f} Hz")
    
    if 'error' in result:
        lines.append("\n" + "-" * 70)
        lines.append(f"ERROR: {result['error']}")
    
    lines.append("\n" + "=" * 70)
    
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("Testing fitnessFunc_claude...")
    print("=" * 70)
    
    # Test with synthetic data
    np.random.seed(42)
    
    # Create mock simulation data
    mock_sim = {
        'spkts': np.random.uniform(0, 1000, 500),  # ms
        'spkids': np.random.randint(0, 10, 500),
        'simDuration': 1000.0,
    }
    
    # Path to experimental data
    exp_path = '/pscratch/sd/k/ktub1999/networkSimulatons_Sonnet/experimental_data.h5'
    
    if os.path.exists(exp_path):
        print(f"\nTesting with experimental data: {exp_path}")
        
        # Compute fitness
        result = fitnessFunc_claude(mock_sim, exp_path)
        
        # Print report
        print(fitness_report(result))
        
        print("\n✓ fitnessFunc_claude working!")
    else:
        print(f"\nExperimental data not found at {exp_path}")
        print("Skipping integration test.")
        
        # Test with minimal mock data
        print("\nTesting feature extraction with mock data...")
        features = extract_simulated_features(mock_sim)
        print(f"  Extracted {features['n_units']} units, {features['total_spikes']} spikes")
        print("\n✓ Feature extraction working!")
