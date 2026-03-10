"""
Fitness Function v2 — Experimental-H5-Aware
=============================================

Works with experimental_features.h5 (create_experimental_target.py output)
and uses parameter_free_burst_detector.compute_network_bursts for hierarchical
burst analysis.

Features:
  • Loads experimental data from the new HDF5 format (unit_*/spike_times + attrs)
  • Extracts simulated features from NetPyNE simData dict
  • Computes per-unit metrics: firing rate, spike count, CV-ISI, E/I ratio
  • Computes hierarchical burst metrics via compute_network_bursts
  • Computes synchrony metrics (sync_spike windows, pairwise correlation)
  • Weighted fitness score (lower is better)
  • Saves fitness.json with full breakdown
  • Plots raster + network activity + burst overlays

Author: Auto-generated
Date: February 2026
"""

import os
import sys
import json
import time
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import h5py

# Ensure the parameter_free_burst_detector is importable
_script_dir = Path(__file__).resolve().parent
_mea_ipn_dir = Path("/pscratch/sd/k/ktub1999/networkSimulations/MEA_Analysis/IPNAnalysis")
for _d in [_script_dir, _mea_ipn_dir]:
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from parameter_free_burst_detector import compute_network_bursts

# Optional imports
try:
    from netpyne import sim
except ImportError:
    sim = None

try:
    from fitness_schema.schema_v2 import (
        DEFAULT_CONFIG,
        fit_schema,
        get_component_weights,
        normalize_schema_weights,
    )
except ImportError:
    try:
        # Fallback: direct relative import
        _schema_dir = _script_dir.parent / "fitness_schema"
        if str(_schema_dir) not in sys.path:
            sys.path.insert(0, str(_schema_dir))
        from schema_v2 import (
            DEFAULT_CONFIG,
            fit_schema,
            get_component_weights,
            normalize_schema_weights,
        )
    except ImportError:
        # Minimal fallback defaults
        DEFAULT_CONFIG = {'max_fitness': 1000.0, 'recording_duration': 300.0}
        fit_schema = {}
        get_component_weights = lambda: {
            'unit_metrics': 0.30, 'quality_metrics': 0.0, 'synchrony': 0.10,
            'burstlets': 0.25, 'network_bursts': 0.25, 'superbursts': 0.10,
        }
        normalize_schema_weights = lambda: fit_schema

# matplotlib — imported lazily in plot function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# EXPERIMENTAL DATA LOADING (from experimental_features.h5)
# =============================================================================

def load_experimental_h5(h5_path: str) -> Dict[str, Any]:
    """
    Load experimental data from the HDF5 file produced by create_experimental_target.py.

    HDF5 structure:
        unit_{id}/spike_times   — dataset  (array of spike times in seconds)
        unit_{id}.attrs         — num_spikes, firing_rate, snr, cell_type, ...
        network_results.attrs   — network-level metadata from network_results.json

    Returns
    -------
    exp_data : dict with keys:
        spike_data          — {int(unit_id): np.array}
        unit_attrs          — {int(unit_id): dict of all xlsx attributes}
        network_results     — dict parsed from network_results group attrs
        n_units, total_spikes, recording_duration, mean_firing_rate, unit_ids
    """
    exp_data = {
        'spike_data': {},
        'unit_attrs': {},
        'network_results': {},
        'n_units': 0,
        'total_spikes': 0,
        'recording_duration': 1.0,
        'mean_firing_rate': 0.0,
        'unit_ids': [],
    }

    try:
        with h5py.File(h5_path, 'r') as f:
            # --- per-unit data ---
            for key in f.keys():
                if not key.startswith('unit_'):
                    continue
                uid = int(key.replace('unit_', ''))
                grp = f[key]

                # spike times
                if 'spike_times' in grp:
                    exp_data['spike_data'][uid] = grp['spike_times'][:]
                else:
                    exp_data['spike_data'][uid] = np.array([])

                # attributes (metrics from xlsx)
                attrs = {}
                for attr_name in grp.attrs:
                    val = grp.attrs[attr_name]
                    # Convert 'NaN' strings back to float nan
                    if isinstance(val, str) and val == 'NaN':
                        val = float('nan')
                    attrs[attr_name] = val
                exp_data['unit_attrs'][uid] = attrs

            # --- network results ---
            if 'network_results' in f:
                nr_grp = f['network_results']
                for attr_name in nr_grp.attrs:
                    val = nr_grp.attrs[attr_name]
                    # Try to JSON-decode complex attrs
                    if isinstance(val, str):
                        try:
                            val = json.loads(val)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    exp_data['network_results'][attr_name] = val

            # --- derived aggregates ---
            exp_data['unit_ids'] = sorted(exp_data['spike_data'].keys())
            exp_data['n_units'] = len(exp_data['unit_ids'])
            exp_data['total_spikes'] = sum(
                len(s) for s in exp_data['spike_data'].values()
            )
            all_spikes = np.concatenate(
                [s for s in exp_data['spike_data'].values() if len(s) > 0]
            ) if exp_data['total_spikes'] > 0 else np.array([])
            exp_data['recording_duration'] = (
                float(all_spikes.max() - all_spikes.min()) if len(all_spikes) > 0 else 1.0
            )
            firing_rates = [
                exp_data['unit_attrs'][uid].get('firing_rate', 0.0)
                for uid in exp_data['unit_ids']
            ]
            exp_data['mean_firing_rate'] = float(np.nanmean(firing_rates)) if firing_rates else 0.0

    except Exception as e:
        logger.error(f"Error loading experimental H5 from {h5_path}: {e}")
        traceback.print_exc()

    return exp_data


# =============================================================================
# SIMULATED DATA LOADING / EXTRACTION
# =============================================================================

def get_sim_data_from_call_stack(kwargs: Dict) -> Dict:
    """Load simulated data from NetPyNE batch call stack."""
    if sim is None:
        raise ImportError("netpyne not available")
    from .utils.utils_old.extract_simulated_data import get_candidate_and_job_path_from_call_stack
    # from .utils.utils_old.batch_helper import get_candidate_and_job_path_from_call_stack
    candidate_path, candidate_label = get_candidate_and_job_path_from_call_stack()
    pkl_path = f'{candidate_path}_data.pkl'
    sim.load(pkl_path)
    kwargs.update({
        'simData': sim.allSimData.todict().copy(),
        'cellData': sim.net.allCells.copy(),
        'popData': sim.net.allPops.copy(),
        'simCfg': sim.cfg.todict().copy(),
        'netParams': sim.net.params.todict().copy(),
        'simLabel': candidate_label,
        'data_file_path': pkl_path,
        'candidate_path': candidate_path,
        'fitness_save_path': f'{candidate_path}_fitness.json',
        'metrics_save_path': f'{candidate_path}_metrics.npy',
    })
    return kwargs


def get_sim_data_from_pkl(kwargs: Dict) -> Dict:
    """Load simulated data from a .pkl path stored in kwargs['sim_data_path']."""
    if sim is None:
        raise ImportError("netpyne not available")
    sim_data_path = kwargs['sim_data_path']
    sim.load(sim_data_path)
    candidate_path = sim_data_path.replace('_data.pkl', '')
    kwargs.update({
        'simData': sim.allSimData.todict().copy(),
        'cellData': sim.net.allCells.copy(),
        'popData': sim.net.allPops.copy(),
        'simCfg': sim.cfg.todict().copy(),
        'netParams': sim.net.params.todict().copy(),
        'simLabel': os.path.basename(candidate_path),
        'data_file_path': sim_data_path,
        'candidate_path': candidate_path,
        'fitness_save_path': f'{candidate_path}_fitness.json',
        'metrics_save_path': f'{candidate_path}_metrics.npy',
    })
    return kwargs


def extract_simulated_features(
    simulated_data: Any,
    recording_duration: Optional[float] = None,
    min_time: Optional[float] = None,
) -> Dict:
    """
    Extract spike_data, firing_rates, cv_isi from NetPyNE simData dict.

    Parameters
    ----------
    simulated_data : dict
        NetPyNE simData dictionary with 'spkt' and 'spkid'.
    recording_duration : float, optional
        Duration in seconds.
    min_time : float, optional
        Exclude spikes before this time (seconds) and shift remaining to t=0.

    Returns
    -------
    features : dict
    """
    spike_data: Dict[int, np.ndarray] = {}

    try:
        if isinstance(simulated_data, dict):
            if 'simData' in simulated_data:
                sd = simulated_data['simData']
            else:
                sd = simulated_data
            spkt = np.array(sd.get('spkt', sd.get('spkts', [])))
            spkid = np.array(sd.get('spkid', sd.get('spkids', [])))
            if recording_duration is None:
                recording_duration = sd.get('simDuration', sd.get('T', 1000.0)) / 1000.0
        elif hasattr(simulated_data, 'allSimData'):
            spkt = np.array(simulated_data.allSimData.get('spkt', []))
            spkid = np.array(simulated_data.allSimData.get('spkid', []))
            if recording_duration is None:
                recording_duration = getattr(
                    getattr(simulated_data, 'cfg', None), 'duration', 1000.0
                ) / 1000.0
        else:
            spkt, spkid = np.array([]), np.array([])
            recording_duration = recording_duration or 1.0

        # Build per-unit spike dicts (ms → s, filter, shift)
        if len(spkt) > 0 and len(spkid) > 0:
            for gid in np.unique(spkid):
                spikes_s = spkt[spkid == gid] / 1000.0
                if min_time is not None and min_time > 0:
                    spikes_s = spikes_s[spikes_s >= min_time] - min_time
                spike_data[int(gid)] = spikes_s

        # Firing rates
        firing_rates = {
            uid: len(sp) / recording_duration if recording_duration > 0 else 0.0
            for uid, sp in spike_data.items()
        }
        # CV-ISI
        cv_isi = {}
        for uid, sp in spike_data.items():
            if len(sp) > 1:
                isis = np.diff(np.sort(sp))
                mu = np.mean(isis)
                cv_isi[uid] = float(np.std(isis) / mu) if mu > 0 else 0.0
            else:
                cv_isi[uid] = 0.0

        features = {
            'spike_data': spike_data,
            'firing_rates': firing_rates,
            'cv_isi': cv_isi,
            'n_units': len(spike_data),
            'total_spikes': sum(len(s) for s in spike_data.values()),
            'recording_duration': recording_duration,
            'mean_firing_rate': float(np.mean(list(firing_rates.values()))) if firing_rates else 0.0,
            'unit_ids': sorted(spike_data.keys()),
        }

    except Exception as e:
        logger.error(f"Error extracting simulated features: {e}")
        traceback.print_exc()
        features = {
            'spike_data': {}, 'firing_rates': {}, 'cv_isi': {},
            'n_units': 0, 'total_spikes': 0, 'recording_duration': recording_duration or 1.0,
            'mean_firing_rate': 0.0, 'unit_ids': [],
        }

    return features


# =============================================================================
# METRIC COMPUTATION
# =============================================================================

def _safe_stat(arr):
    """Return (mean, std) for array, handles empty."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))


def compute_unit_score(sim_features: Dict, exp_data: Dict, recording_duration: float) -> Dict[str, Any]:
    """Compare per-unit metrics between sim and experiment."""
    max_score = DEFAULT_CONFIG.get('max_fitness', 1000.0)

    # -- firing rates --
    sim_frs = list(sim_features['firing_rates'].values())
    exp_frs = [
        exp_data['unit_attrs'][uid].get('firing_rate', 0.0)
        for uid in exp_data['unit_ids']
    ]
    sim_mean_fr, _ = _safe_stat(sim_frs)
    exp_mean_fr, _ = _safe_stat(exp_frs)
    firing_rate_error = ((sim_mean_fr - exp_mean_fr) ** 2) / max(exp_mean_fr ** 2, 1e-6) * 100.0

    # -- spike counts --
    sim_total = sim_features['total_spikes']
    exp_total = exp_data['total_spikes']
    denom = max(exp_total**2, 1e-6)
    num_spikes_error = ((sim_total - exp_total) ** 2) / denom * 100.0

    # -- firing range --
    sim_fr_range = (max(sim_frs) - min(sim_frs)) if sim_frs else 0.0
    exp_fr_ranges = [
        exp_data['unit_attrs'][uid].get('firing_range', 0.0)
        for uid in exp_data['unit_ids']
    ]
    exp_mean_fr_range, _ = _safe_stat(exp_fr_ranges)
    firing_range_error = abs(sim_fr_range - exp_mean_fr_range) / max(exp_mean_fr_range, 1e-6) * 100.0

    # -- CV ISI --
    sim_cv = list(sim_features['cv_isi'].values())
    sim_mean_cv, _ = _safe_stat(sim_cv)
    # No direct CV-ISI in xlsx; compute from ISI violations as proxy
    exp_cvs = []
    for uid in exp_data['unit_ids']:
        attrs = exp_data['unit_attrs'][uid]
        iv = attrs.get('isi_violations_ratio', 0.0)
        if isinstance(iv, str):
            iv = 0.0
        exp_cvs.append(float(iv) if np.isfinite(float(iv)) else 0.0)
    exp_mean_cv, _ = _safe_stat(exp_cvs)
    cv_isi_error = abs(sim_mean_cv - exp_mean_cv) / max(exp_mean_cv, 1e-6) * 100.0

    # -- n_units --
    n_units_error = abs(sim_features['n_units'] - exp_data['n_units']) / max(exp_data['n_units'], 1) * 100.0

    # -- E/I ratio --
    exp_types = [
        exp_data['unit_attrs'][uid].get('cell_type', 'excitatory')
        for uid in exp_data['unit_ids']
    ]
    exp_inh_frac = sum(1 for t in exp_types if t == 'inhibitory') / max(len(exp_types), 1)
    # For simulated: classify top-20% as inhibitory
    if sim_frs:
        thr = np.percentile(sim_frs, 80)
        sim_inh_frac = sum(1 for fr in sim_frs if fr >= thr) / len(sim_frs)
    else:
        sim_inh_frac = 0.0
    ei_ratio_error = abs(sim_inh_frac - exp_inh_frac) / max(exp_inh_frac, 1e-6) * 100.0

    # -- weighted total --
    schema = fit_schema.get('unit_metrics', {}).get('metrics', {})
    def _w(name):
        return schema.get(name, {}).get('weight', 1.0)
    def _clip(val, name):
        mx = schema.get(name, {}).get('max_val', max_score)
        return min(val, mx)

    total = (
        _w('firing_rate_error')   * _clip(firing_rate_error,   'firing_rate_error') +
        _w('num_spikes_error')    * _clip(num_spikes_error,    'num_spikes_error') +
        _w('firing_range_error')  * _clip(firing_range_error,  'firing_range_error') +
        _w('cv_isi_error')        * _clip(cv_isi_error,        'cv_isi_error') +
        _w('n_units_error')       * _clip(n_units_error,       'n_units_error') +
        _w('ei_ratio_error')      * _clip(ei_ratio_error,      'ei_ratio_error')
    )

    return {
        'firing_rate_error': float(firing_rate_error),
        'sim_mean_firing_rate': float(sim_mean_fr),
        'exp_mean_firing_rate': float(exp_mean_fr),
        'num_spikes_error': float(num_spikes_error),
        'sim_total_spikes': int(sim_total),
        'exp_total_spikes': int(exp_total),
        'firing_range_error': float(firing_range_error),
        'sim_firing_range': float(sim_fr_range),
        'exp_mean_firing_range': float(exp_mean_fr_range),
        'cv_isi_error': float(cv_isi_error),
        'sim_mean_cv_isi': float(sim_mean_cv),
        'exp_mean_cv_isi': float(exp_mean_cv),
        'n_units_error': float(n_units_error),
        'sim_n_units': int(sim_features['n_units']),
        'exp_n_units': int(exp_data['n_units']),
        'ei_ratio_error': float(ei_ratio_error),
        'sim_inh_fraction': float(sim_inh_frac),
        'exp_inh_fraction': float(exp_inh_frac),
        'total_score': float(total),
    }


def compute_synchrony_score(
    sim_spike_data: Dict[int, np.ndarray],
    exp_data: Dict,
    bin_size: float = 0.01,
    recording_duration: float = 1.0,
) -> Dict[str, Any]:
    """Compare synchrony metrics."""
    # -- sync_spike_N from experimental attrs --
    def _mean_sync(attr_name):
        vals = [
            exp_data['unit_attrs'][uid].get(attr_name, 0.0)
            for uid in exp_data['unit_ids']
        ]
        vals = [float(v) for v in vals if not (isinstance(v, str) or (isinstance(v, float) and np.isnan(v)))]
        return float(np.mean(vals)) if vals else 0.0

    exp_sync2 = _mean_sync('sync_spike_2')
    exp_sync4 = _mean_sync('sync_spike_4')
    exp_sync8 = _mean_sync('sync_spike_8')

    # Simulated synchrony: fraction of spike pairs within window
    def _sim_sync(spike_data, window_ms):
        window_s = window_ms / 1000.0
        if len(spike_data) < 2:
            return 0.0
        all_spikes = np.sort(np.concatenate(list(spike_data.values())))
        if len(all_spikes) < 2:
            return 0.0
        diffs = np.diff(all_spikes)
        return float(np.mean(diffs < window_s))

    sim_sync2 = _sim_sync(sim_spike_data, 2)
    sim_sync4 = _sim_sync(sim_spike_data, 4)
    sim_sync8 = _sim_sync(sim_spike_data, 8)

    sync2_err = abs(sim_sync2 - exp_sync2)
    sync4_err = abs(sim_sync4 - exp_sync4)
    sync8_err = abs(sim_sync8 - exp_sync8)

    # Pairwise correlation (binned)
    def _pairwise_corr(spike_data, bin_sz, dur):
        if len(spike_data) < 2:
            return 0.0
        uids = sorted(spike_data.keys())[:50]  # cap for speed
        n_bins = max(1, int(dur / bin_sz))
        bins_arr = np.linspace(0, dur, n_bins + 1)
        mat = np.zeros((len(uids), n_bins))
        for i, uid in enumerate(uids):
            counts, _ = np.histogram(spike_data[uid], bins=bins_arr)
            mat[i] = counts
        cc = np.corrcoef(mat)
        mask = np.triu(np.ones_like(cc, dtype=bool), k=1)
        vals = cc[mask]
        vals = vals[np.isfinite(vals)]
        return float(np.mean(vals)) if len(vals) > 0 else 0.0

    sim_corr = _pairwise_corr(sim_spike_data, bin_size, recording_duration)
    exp_corr = _pairwise_corr(exp_data['spike_data'], bin_size, recording_duration)
    corr_err = abs(sim_corr - exp_corr)

    schema = fit_schema.get('synchrony', {}).get('metrics', {})
    def _w(n):
        return schema.get(n, {}).get('weight', 0.25)

    total = (
        _w('sync_spike_2_error')          * min(sync2_err, 1.0) +
        _w('sync_spike_4_error')          * min(sync4_err, 1.0) +
        _w('sync_spike_8_error')          * min(sync8_err, 1.0) +
        _w('pairwise_correlation_error')  * min(corr_err, 1.0)
    )

    return {
        'sync_spike_2_error': float(sync2_err),
        'sync_spike_4_error': float(sync4_err),
        'sync_spike_8_error': float(sync8_err),
        'pairwise_correlation_error': float(corr_err),
        'sim_sync2': float(sim_sync2),
        'sim_sync4': float(sim_sync4),
        'sim_sync8': float(sim_sync8),
        'sim_pairwise_corr': float(sim_corr),
        'exp_sync2': float(exp_sync2),
        'exp_sync4': float(exp_sync4),
        'exp_sync8': float(exp_sync8),
        'exp_pairwise_corr': float(exp_corr),
        'total_score': float(total),
    }


# =============================================================================
# HIERARCHICAL BURST SCORING  (uses parameter_free_burst_detector)
# =============================================================================

def _victor_purpura_distance(
    train_a: np.ndarray,
    train_b: np.ndarray,
    q: float = 1.0,
) -> float:
    """
    Victor-Purpura spike-train distance.

    Computes the minimal cost to transform *train_a* into *train_b* using
    three elementary operations:

      * Insert a spike  — cost 1
      * Delete a spike  — cost 1
      * Move a spike by Δt — cost q·|Δt|

    A move is cheaper than a delete+insert when q·|Δt| < 2, i.e. when the
    two spikes are closer than 2/q.  The parameter *q* therefore sets the
    temporal precision of the comparison (units: 1/seconds when spike
    times are in seconds).

    Parameters
    ----------
    train_a, train_b : 1-D array-like
        Sorted spike / event times.
    q : float
        Cost per unit time of moving a spike.  Larger *q* emphasises
        temporal precision; *q* → 0 recovers the difference in spike
        counts; *q* → ∞ recovers the number of non-coincident spikes.

    Returns
    -------
    distance : float
        The Victor-Purpura distance (≥ 0).  Zero when the two trains
        are identical.
    """
    a = np.asarray(train_a, dtype=float)
    b = np.asarray(train_b, dtype=float)
    n = len(a)
    m = len(b)

    if n == 0:
        return float(m)
    if m == 0:
        return float(n)

    # DP table — d[i][j] = cost to match a[:i] with b[:j]
    d = np.zeros((n + 1, m + 1), dtype=float)
    d[:, 0] = np.arange(n + 1, dtype=float)   # delete all of a
    d[0, :] = np.arange(m + 1, dtype=float)   # insert all of b

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost_move = d[i - 1, j - 1] + q * abs(a[i - 1] - b[j - 1])
            cost_del  = d[i - 1, j] + 1.0     # delete a[i-1]
            cost_ins  = d[i, j - 1] + 1.0     # insert b[j-1]
            d[i, j] = min(cost_move, cost_del, cost_ins)

    return float(d[n, m])


def _run_burst_detector(spike_data: Dict[int, np.ndarray], **burst_params) -> Dict:
    """Thin wrapper around compute_network_bursts."""
    if not spike_data:
        return {
            'burstlets': {'events': [], 'metrics': {}},
            'network_bursts': {'events': [], 'metrics': {}},
            'superbursts': {'events': [], 'metrics': {}},
            'diagnostics': {},
            'plot_data': {},
        }
    result = compute_network_bursts(SpikeTimes=spike_data, plot=False, verbose=False, **burst_params)
    if 'error' in result:
        logger.warning(f"Burst detector returned error: {result['error']}")
        return {
            'burstlets': {'events': [], 'metrics': {}},
            'network_bursts': {'events': [], 'metrics': {}},
            'superbursts': {'events': [], 'metrics': {}},
            'diagnostics': {},
            'plot_data': {},
        }
    return result


def _score_burst_level(
    sim_events: List[Dict],
    exp_events: List[Dict],
    sim_metrics: Dict,
    exp_metrics: Dict,
    recording_duration: float,
    level_name: str,
) -> Dict[str, Any]:
    """
    Score one hierarchical burst level (burstlets / network_bursts / superbursts).

    Every metric is normalised by its experimental value and expressed as a
    squared relative error::

        metric_error = ((sim_val - exp_val) / max(|exp_val|, 1e-6)) ** 2

    The total score is the **mean squared error** (MSE) across all seven
    per-metric normalised squared errors, scaled by 100 for readability.
    """
    _EPS = 1e-6

    n_sim = len(sim_events)
    n_exp = len(exp_events)

    # -- count --
    count_error = ((n_sim - n_exp) / max(n_exp, _EPS)) ** 2

    # -- rate --
    sim_rate = sim_metrics.get('rate', n_sim / recording_duration if recording_duration > 0 else 0)
    exp_rate = exp_metrics.get('rate', n_exp / recording_duration if recording_duration > 0 else 0)
    rate_error = ((sim_rate - exp_rate) / max(abs(exp_rate), _EPS)) ** 2

    # -- duration --
    sim_dur = sim_metrics.get('duration', {}).get('mean', 0.0)
    exp_dur = exp_metrics.get('duration', {}).get('mean', 0.0)
    duration_error = ((sim_dur - exp_dur) / max(abs(exp_dur), _EPS)) ** 2

    # -- IBI (inter-burst interval) --
    sim_ibi = sim_metrics.get('inter_event_interval', {}).get('mean', 0.0)
    exp_ibi = exp_metrics.get('inter_event_interval', {}).get('mean', 0.0)
    ibi_error = ((sim_ibi - exp_ibi) / max(abs(exp_ibi), _EPS)) ** 2

    # -- participation --
    sim_part = sim_metrics.get('participation', {}).get('mean', 0.0)
    exp_part = exp_metrics.get('participation', {}).get('mean', 0.0)
    participation_error = ((sim_part - exp_part) / max(abs(exp_part), _EPS)) ** 2

    # -- intensity / energy --
    sim_int = sim_metrics.get('intensity', {}).get('mean', 0.0)
    exp_int = exp_metrics.get('intensity', {}).get('mean', 0.0)
    intensity_error = ((sim_int - exp_int) / max(abs(exp_int), _EPS)) ** 2

    # -- spikes per burst --
    sim_spb = sim_metrics.get('spikes_per_burst', {}).get('mean', 0.0)
    exp_spb = exp_metrics.get('spikes_per_burst', {}).get('mean', 0.0)
    spikes_per_burst_error = ((sim_spb - exp_spb) / max(abs(exp_spb), _EPS)) ** 2

    # -- MSE across all normalised squared errors (×100 for readability) --
    all_errors = [
        count_error, rate_error, duration_error, ibi_error,
        participation_error, intensity_error, spikes_per_burst_error,
    ]
    total = float(np.mean(all_errors)) * 100.0

    return {
        'burst_count_error': float(count_error),
        'burst_rate_error': float(rate_error),
        'duration_error': float(duration_error),
        'ibi_error': float(ibi_error),
        'participation_error': float(participation_error),
        'intensity_error': float(intensity_error),
        'spikes_per_burst_error': float(spikes_per_burst_error),
        'n_sim': n_sim,
        'n_exp': n_exp,
        'sim_rate': float(sim_rate),
        'exp_rate': float(exp_rate),
        'sim_duration': float(sim_dur),
        'exp_duration': float(exp_dur),
        'sim_ibi': float(sim_ibi),
        'exp_ibi': float(exp_ibi),
        'sim_participation': float(sim_part),
        'exp_participation': float(exp_part),
        'sim_intensity': float(sim_int),
        'exp_intensity': float(exp_int),
        'sim_spikes_per_burst': float(sim_spb),
        'exp_spikes_per_burst': float(exp_spb),
        'total_score': float(total),
    }


def compute_hierarchical_burst_scores(
    sim_spike_data: Dict[int, np.ndarray],
    exp_spike_data: Dict[int, np.ndarray],
    recording_duration: float,
    burst_params: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Run parameter_free_burst_detector on both sim and exp, then score
    each hierarchical level.

    Uses base_threshold_multiplier=45 and min_burstlet_participation=0.05
    for the main burst detection, and base_threshold_multiplier=15 with
    min_burstlet_participation=0.05 for pre-burstlet detection.
    """
    bp = burst_params or {}

    # Main burst detection (threshold=45)
    sim_bursts = _run_burst_detector(
        sim_spike_data, min_burstlet_participation=0.05,
        base_threshold_multiplier=40, **bp,
    )
    exp_bursts = _run_burst_detector(
        exp_spike_data, min_burstlet_participation=0.05,
        base_threshold_multiplier=40, **bp,
    )

    results = {}
    for level in ['burstlets', 'network_bursts', 'superbursts']:
        sim_evts = sim_bursts.get(level, {}).get('events', [])
        exp_evts = exp_bursts.get(level, {}).get('events', [])
        sim_met  = sim_bursts.get(level, {}).get('metrics', {})
        exp_met  = exp_bursts.get(level, {}).get('metrics', {})
        results[level] = _score_burst_level(
            sim_evts, exp_evts, sim_met, exp_met, recording_duration, level,
        )

    # ------------------------------------------------------------------
    # Pre-burstlet score (lower threshold = 15×)
    # ------------------------------------------------------------------
    sim_pre_bursts = _run_burst_detector(
        sim_spike_data, base_threshold_multiplier=15,
        min_burstlet_participation=0.05,
    )
    exp_pre_bursts = _run_burst_detector(
        exp_spike_data, base_threshold_multiplier=15,
        min_burstlet_participation=0.05,
    )

    sim_pre_evts = sim_pre_bursts.get('burstlets', {}).get('events', [])
    exp_pre_evts = exp_pre_bursts.get('burstlets', {}).get('events', [])
    sim_pre_met  = sim_pre_bursts.get('burstlets', {}).get('metrics', {})
    exp_pre_met  = exp_pre_bursts.get('burstlets', {}).get('metrics', {})

    results['pre_burstlets'] = _score_burst_level(
        sim_pre_evts, exp_pre_evts, sim_pre_met, exp_pre_met,
        recording_duration, 'pre_burstlets',
    )

    sim_pre_threshold = sim_pre_bursts.get('plot_data', {}).get('threshold', None)
    exp_pre_threshold = exp_pre_bursts.get('plot_data', {}).get('threshold', None)
    results['pre_burstlets']['sim_threshold'] = sim_pre_threshold
    results['pre_burstlets']['exp_threshold'] = exp_pre_threshold

    # --- Diagnostics ---
    results['diagnostics'] = {
        'sim': sim_bursts.get('diagnostics', {}),
        'exp': exp_bursts.get('diagnostics', {}),
    }

    # --- Plot data (merge pre-burst info) ---
    sim_pre_pd = sim_pre_bursts.get('plot_data', {})
    exp_pre_pd = exp_pre_bursts.get('plot_data', {})
    results['sim_plot_data'] = {
        **sim_bursts.get('plot_data', {}),
        'pre_burst_threshold': sim_pre_threshold,
        'pre_burst_peak_times': sim_pre_pd.get('burst_peak_times', None),
        'pre_burst_peak_values': sim_pre_pd.get('burst_peak_values', None),
    }
    results['exp_plot_data'] = {
        **exp_bursts.get('plot_data', {}),
        'pre_burst_threshold': exp_pre_threshold,
        'pre_burst_peak_times': exp_pre_pd.get('burst_peak_times', None),
        'pre_burst_peak_values': exp_pre_pd.get('burst_peak_values', None),
    }

    # Keep raw events for plotting
    results['sim_burst_events'] = {
        level: sim_bursts.get(level, {}).get('events', [])
        for level in ['burstlets', 'network_bursts', 'superbursts']
    }
    results['exp_burst_events'] = {
        level: exp_bursts.get(level, {}).get('events', [])
        for level in ['burstlets', 'network_bursts', 'superbursts']
    }

    return results


def compute_hierarchical_burst_scores_simple(
    sim_spike_data: Dict[int, np.ndarray],
    exp_spike_data: Dict[int, np.ndarray],
    recording_duration: float,
    burst_params: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Simplified hierarchical burst scoring.

    Runs parameter_free_burst_detector on both sim and exp spike data,
    counts the number of events at each hierarchical level (burstlets,
    network_bursts, superbursts), and returns the MSE of those counts
    as the overall score.

    Returns
    -------
    results : dict with keys:
        burstlets / network_bursts / superbursts — per-level dicts:
            n_sim, n_exp, squared_error
        mse        — mean squared error across the three levels
        total_score — same as mse (lower is better)
        sim_burst_events / exp_burst_events — raw event lists for plotting
    """
    bp = burst_params or {}

    sim_bursts = _run_burst_detector(sim_spike_data, min_burstlet_participation=0.05, **bp)
    exp_bursts = _run_burst_detector(exp_spike_data, min_burstlet_participation=0.05, **bp)

    levels = ['burstlets', 'network_bursts', 'superbursts']
    results = {}
    squared_errors = []

    for level in levels:
        n_sim = len(sim_bursts.get(level, {}).get('events', []))
        n_exp = len(exp_bursts.get(level, {}).get('events', []))
        se = float((n_sim - n_exp) ** 2)
        squared_errors.append(se)
        results[level] = {
            'n_sim': n_sim,
            'n_exp': n_exp,
            'total_score': se,
        }

    mse = float(np.mean(squared_errors))
    results['mse'] = mse
    results['total_score'] = mse

    # ------------------------------------------------------------------
    # Pre-burstlet score: re-run detector with lower threshold (1.2x)
    # so sub-threshold events that precede true bursts are captured.
    # ------------------------------------------------------------------
    pre_bp = {**bp, 'base_threshold_multiplier': 1.2}
    sim_pre_bursts = _run_burst_detector(sim_spike_data, **pre_bp)
    exp_pre_bursts = _run_burst_detector(exp_spike_data, **pre_bp)
    n_sim_pre = len(sim_pre_bursts.get('burstlets', {}).get('events', []))
    n_exp_pre = len(exp_pre_bursts.get('burstlets', {}).get('events', []))
    se_pre = float((n_sim_pre - n_exp_pre) ** 2)

    sim_pre_threshold = sim_pre_bursts.get('plot_data', {}).get('threshold', None)
    exp_pre_threshold = exp_pre_bursts.get('plot_data', {}).get('threshold', None)

    results['pre_burstlets'] = {
        'n_sim': n_sim_pre,
        'n_exp': n_exp_pre,
        'total_score': se_pre,
        'sim_threshold': sim_pre_threshold,
        'exp_threshold': exp_pre_threshold,
    }

    results['sim_burst_events'] = {
        level: sim_bursts.get(level, {}).get('events', [])
        for level in levels
    }
    results['exp_burst_events'] = {
        level: exp_bursts.get(level, {}).get('events', [])
        for level in levels
    }
    results['diagnostics'] = {
        'sim': sim_bursts.get('diagnostics', {}),
        'exp': exp_bursts.get('diagnostics', {}),
    }
    results['sim_plot_data'] = {**sim_bursts.get('plot_data', {}), 'pre_burst_threshold': sim_pre_threshold}
    results['exp_plot_data'] = {**exp_bursts.get('plot_data', {}), 'pre_burst_threshold': exp_pre_threshold}
    return results


def compute_hierarchical_burst_scores_simple_v2(
    sim_spike_data: Dict[int, np.ndarray],
    exp_spike_data: Dict[int, np.ndarray],
    recording_duration: float,
    burst_params: Optional[Dict] = None,
    level_weights: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """
    Simplified hierarchical burst scoring **v2**.

    For each hierarchical level (burstlets, network_bursts, superbursts) this
    computes two sub-scores and combines them with configurable weights:

      1. **count_score** — squared error of event counts:  (n_sim − n_exp)²
      2. **timing_score** — MSE of matched event start-times.  Both event
         lists are sorted by ``start``; the shorter list length determines
         the number of matched pairs, and the MSE is computed over those
         pairs.  If either side has zero events the timing score is 0
         (the count score already penalises the mismatch).

    The per-level score is::

        level_score = w_count * count_score + w_timing * timing_score

    and the overall ``total_score`` is the mean of the three level scores.

    Parameters
    ----------
    sim_spike_data, exp_spike_data : dict
        {unit_id: np.ndarray of spike times (seconds)}.
    recording_duration : float
        Duration of the recording in seconds.
    burst_params : dict, optional
        Extra kwargs forwarded to ``compute_network_bursts``.
    level_weights : dict, optional
        Per-level sub-score weights, e.g.::

            {
                'burstlets':      {'count': 0.6, 'timing': 0.4},
                'network_bursts': {'count': 0.5, 'timing': 0.5},
                'superbursts':    {'count': 0.7, 'timing': 0.3},
                'pre_burstlets':  {'count': 0.5, 'timing': 0.5},
            }

        Defaults to ``{'count': 0.5, 'timing': 0.5}`` for every level.

    Returns
    -------
    results : dict
        Same top-level keys as ``compute_hierarchical_burst_scores_simple``
        (burstlets, network_bursts, superbursts, pre_burstlets, mse,
        total_score, sim/exp_burst_events, diagnostics, sim/exp_plot_data)
        but each level dict now also contains ``count_score``,
        ``timing_score``, and the ``weights`` that were used.
    """
    bp = burst_params or {}

    sim_bursts = _run_burst_detector(sim_spike_data, min_burstlet_participation=0.05,base_threshold_multiplier = 45, **bp)
    exp_bursts = _run_burst_detector(exp_spike_data, base_threshold_multiplier=45, **bp)

    # --- Early exit: penalise simulations with a superburst in the first 2.5 s ---
    _max_score = DEFAULT_CONFIG.get('max_fitness', 30000.0)
    _early_superbursts = [
        ev for ev in sim_bursts.get('superbursts', {}).get('events', [])
        if ev.get('start', float('inf')) < 2.5
    ]
    if _early_superbursts:
        logger.warning(
            f"Superburst detected in first 2.5 s (start={_early_superbursts[0]['start']:.2f} s) — "
            "returning max_score for all levels."
        )
        _levels = ['burstlets', 'network_bursts', 'superbursts']
        _r = {
            level: {
                'n_sim': len(sim_bursts.get(level, {}).get('events', [])),
                'n_exp': len(exp_bursts.get(level, {}).get('events', [])),
                'count_score': _max_score,
                'timing_score': _max_score,
                'total_score': _max_score,
                'weights': {'count': 0.5, 'timing': 0.5},
            }
            for level in _levels
        }
        _r['pre_burstlets'] = {
            'n_sim': 0, 'n_exp': 0,
            'count_score': _max_score, 'timing_score': _max_score,
            'total_score': _max_score,
            'weights': {'count': 0.5, 'timing': 0.5},
            'sim_threshold': None, 'exp_threshold': None,
        }
        _r['mse'] = _max_score
        _r['total_score'] = _max_score
        _r['sim_burst_events'] = {lv: sim_bursts.get(lv, {}).get('events', []) for lv in _levels}
        _r['exp_burst_events'] = {lv: exp_bursts.get(lv, {}).get('events', []) for lv in _levels}
        _r['diagnostics'] = {
            'sim': sim_bursts.get('diagnostics', {}),
            'exp': exp_bursts.get('diagnostics', {}),
        }
        _r['sim_plot_data'] = sim_bursts.get('plot_data', {})
        _r['exp_plot_data'] = exp_bursts.get('plot_data', {})
        return _r

    default_w = {'count': 0.5, 'timing': 0.5}
    lw = level_weights or {}

    levels = ['burstlets', 'network_bursts', 'superbursts']
    results = {}
    level_scores = []

    for level in levels:
        sim_events = sim_bursts.get(level, {}).get('events', [])
        exp_events = exp_bursts.get(level, {}).get('events', [])
        n_sim = len(sim_events)
        n_exp = len(exp_events)

        # --- Count score: squared error ---
        count_se = abs(n_sim - n_exp)/max(n_exp, 1)

        # --- Timing score: Victor-Purpura distance on start times ---
        #     q = 1.0  →  move cheaper than del+ins when |Δt| < 2 s
        sim_starts = np.array(sorted(ev['start'] for ev in sim_events))
        exp_starts = np.array(sorted(ev['start'] for ev in exp_events))
        timing_vp = _victor_purpura_distance(sim_starts, exp_starts, q=1.0)/max(n_exp, 1)
        # if n_sim > 0:
        #     timing_vp = _victor_purpura_distance(sim_starts, exp_starts, q=1.0)/max(n_exp, 1)
        # else:
        #     timing_vp = 1000.0

        # --- Combined score for this level ---
        w = lw.get(level, default_w)
        w_count  = w.get('count',  0.5)
        w_timing = w.get('timing', 0.5)
        level_score = (w_count * count_se + w_timing * timing_vp)*100

        results[level] = {
            'n_sim': n_sim,
            'n_exp': n_exp,
            'count_score': count_se,
            'timing_score': timing_vp,
            'total_score': level_score,
            'weights': {'count': w_count, 'timing': w_timing},
        }
        level_scores.append(level_score)

    mse = float(np.mean(level_scores))
    results['mse'] = mse
    results['total_score'] = mse

    # ------------------------------------------------------------------
    # Pre-burstlet score (lower threshold = 1.2×)
    # ------------------------------------------------------------------
    # pre_bp = {**bp, 'base_threshold_multiplier': 1.2}
    sim_pre_bursts = _run_burst_detector(sim_spike_data,base_threshold_multiplier= 15,min_burstlet_participation=0.05 )
    exp_pre_bursts = _run_burst_detector(exp_spike_data ,base_threshold_multiplier= 15,min_burstlet_participation=0.05)

    sim_pre_events = sim_pre_bursts.get('burstlets', {}).get('events', [])
    exp_pre_events = exp_pre_bursts.get('burstlets', {}).get('events', [])
    n_sim_pre = len(sim_pre_events)
    n_exp_pre = len(exp_pre_events)

    pre_count_se = abs(n_sim_pre - n_exp_pre)/max(n_exp_pre, 1)

    sim_pre_starts = np.array(sorted(ev['start'] for ev in sim_pre_events))
    exp_pre_starts = np.array(sorted(ev['start'] for ev in exp_pre_events))
    pre_timing_vp = _victor_purpura_distance(sim_pre_starts, exp_pre_starts, q=1.0)/max(n_exp_pre, 1)
    # if n_sim_pre > 0:
    #     pre_timing_vp = _victor_purpura_distance(sim_pre_starts, exp_pre_starts, q=1.0)/max(n_exp_pre, 1)
    # else:
    #     pre_timing_vp = 10000.0

    pre_w = lw.get('pre_burstlets', default_w)
    pre_score = (pre_w.get('count', 0.5) * pre_count_se + pre_w.get('timing', 0.5) * pre_timing_vp)*100

    sim_pre_threshold = sim_pre_bursts.get('plot_data', {}).get('threshold', None)
    exp_pre_threshold = exp_pre_bursts.get('plot_data', {}).get('threshold', None)

    results['pre_burstlets'] = {
        'n_sim': n_sim_pre,
        'n_exp': n_exp_pre,
        'count_score': pre_count_se,
        'timing_score': pre_timing_vp,
        'total_score': pre_score,
        'weights': {'count': pre_w.get('count', 0.5), 'timing': pre_w.get('timing', 0.5)},
        'sim_threshold': sim_pre_threshold,
        'exp_threshold': exp_pre_threshold,
    }

    # --- Raw events, diagnostics, plot data ---
    results['sim_burst_events'] = {
        level: sim_bursts.get(level, {}).get('events', [])
        for level in levels
    }
    results['exp_burst_events'] = {
        level: exp_bursts.get(level, {}).get('events', [])
        for level in levels
    }
    results['diagnostics'] = {
        'sim': sim_bursts.get('diagnostics', {}),
        'exp': exp_bursts.get('diagnostics', {}),
    }
    sim_pre_pd = sim_pre_bursts.get('plot_data', {})
    exp_pre_pd = exp_pre_bursts.get('plot_data', {})
    results['sim_plot_data'] = {
        **sim_bursts.get('plot_data', {}),
        'pre_burst_threshold': sim_pre_threshold,
        'pre_burst_peak_times': sim_pre_pd.get('burst_peak_times', None),
        'pre_burst_peak_values': sim_pre_pd.get('burst_peak_values', None),
    }
    results['exp_plot_data'] = {
        **exp_bursts.get('plot_data', {}),
        'pre_burst_threshold': exp_pre_threshold,
        'pre_burst_peak_times': exp_pre_pd.get('burst_peak_times', None),
        'pre_burst_peak_values': exp_pre_pd.get('burst_peak_values', None),
    }
    return results


# =============================================================================
# PLOTTING
# =============================================================================

def plot_fitness_comparison(
    sim_features: Dict,
    exp_data: Dict,
    burst_results: Dict,
    save_path: str,
    title_prefix: str = "",
):
    """
    Generate a multi-panel plot:
        Row 1: Simulated raster
        Row 2: Simulated network activity + burst overlays
        Row 3: Experimental raster
        Row 4: Experimental network activity + burst overlays
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    ax_sim_raster, ax_sim_net, ax_exp_raster, ax_exp_net = axes

    # --- Build cell_types for simulated data (top 20% FR = inhibitory) ---
    sim_cell_types = {}
    sim_frs = sim_features.get('firing_rates', {})
    if sim_frs:
        fr_values = list(sim_frs.values())
        thr = np.percentile(fr_values, 80) if fr_values else 0.0
        for uid, fr in sim_frs.items():
            sim_cell_types[uid] = 'inhibitory' if fr >= thr else 'excitatory'

    # --- Build cell_types for experimental data ---
    exp_cell_types = {}
    for uid in exp_data.get('unit_ids', []):
        ct = exp_data.get('unit_attrs', {}).get(uid, {}).get('cell_type', 'excitatory')
        exp_cell_types[uid] = ct
    
    sim_cell_types = exp_cell_types


    # --- Sim raster ---
    _plot_raster(ax_sim_raster, sim_features['spike_data'],
                 title=f"{title_prefix}Simulated Raster",
                 cell_types=sim_cell_types)
    # Overlay burst regions on simulated raster
    _overlay_bursts(ax_sim_raster, burst_results.get('sim_burst_events', {}))

    # --- Sim network activity ---
    sim_pd = burst_results.get('sim_plot_data', {})
    if sim_pd:
        _plot_network_signal(ax_sim_net, sim_pd, title="Simulated Network Activity")
        _overlay_bursts(ax_sim_net, burst_results.get('sim_burst_events', {}))
    else:
        ax_sim_net.text(0.5, 0.5, "No sim burst data", transform=ax_sim_net.transAxes, ha='center')

    # --- Exp raster ---
    _plot_raster(ax_exp_raster, exp_data['spike_data'],
                 title=f"{title_prefix}Experimental Raster",
                 cell_types=exp_cell_types)
    # Overlay burst regions on experimental raster
    _overlay_bursts(ax_exp_raster, burst_results.get('exp_burst_events', {}))

    # --- Exp network activity ---
    exp_pd = burst_results.get('exp_plot_data', {})
    if exp_pd:
        _plot_network_signal(ax_exp_net, exp_pd, title="Experimental Network Activity")
        _overlay_bursts(ax_exp_net, burst_results.get('exp_burst_events', {}))
    else:
        ax_exp_net.text(0.5, 0.5, "No exp burst data", transform=ax_exp_net.transAxes, ha='center')

    ax_exp_net.set_xlabel("Time (s)")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.08)

    # Save full, then zoomed versions
    fig.savefig(save_path, dpi=200)
    logger.info(f"Saved plot: {save_path}")

    # 60s zoom
    for ax in axes:
        ax.set_xlim(0, 60)
    zoom_path = save_path.replace('.png', '_60s.png').replace('.svg', '_60s.svg')
    fig.savefig(zoom_path, dpi=200)

    # 30s zoom
    for ax in axes:
        ax.set_xlim(0, 30)
    zoom_path2 = save_path.replace('.png', '_30s.png').replace('.svg', '_30s.svg')
    fig.savefig(zoom_path2, dpi=200)

    plt.close(fig)


def _plot_raster(ax, spike_data, title="", cell_types=None):
    """Clean raster plot — one row per unit, color-coded by cell type."""
    units = sorted(spike_data.keys())
    for y, uid in enumerate(units):
        times = spike_data[uid]
        if len(times) == 0:
            continue
        if cell_types is not None:
            ct = cell_types.get(uid, 'excitatory')
            color = 'red' if ct == 'inhibitory' else 'blue'
        else:
            color = 'gray'
        ax.plot(
            times, np.full_like(times, y),
            linestyle='None', marker='|', markersize=3,
            markeredgewidth=0.5, color=color, alpha=0.8, rasterized=True,
        )
    ax.set_ylabel("Unit Index")
    ax.set_ylim(-1, len(units))
    ax.set_title(title, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend for cell types and burst overlays
    if cell_types is not None:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        handles = [
            Line2D([0], [0], marker='|', color='blue', linestyle='None',
                   markersize=6, markeredgewidth=1.5, label='Excitatory'),
            Line2D([0], [0], marker='|', color='red', linestyle='None',
                   markersize=6, markeredgewidth=1.5, label='Inhibitory'),
            Patch(facecolor='tab:green', alpha=0.3, label='Burstlets'),
            Patch(facecolor='tab:blue', alpha=0.3, label='Network Bursts'),
            Patch(facecolor='purple', alpha=0.3, label='Superbursts'),
        ]
        ax.legend(handles=handles, loc='upper right', fontsize=7,
                  ncol=5, framealpha=0.7, handlelength=1.5)


def _plot_network_signal(ax, plot_data, title=""):
    """Plot synchrony signal with baseline / threshold."""
    t = plot_data.get('t', np.array([]))
    signal = plot_data.get('signal', np.array([]))
    signal_smooth = plot_data.get('signal_smooth', None)
    baseline = plot_data.get('baseline', None)
    threshold = plot_data.get('threshold', None)
    pre_burst_threshold = plot_data.get('pre_burst_threshold', None)
    burst_peak_times = plot_data.get('burst_peak_times', None)
    burst_peak_values = plot_data.get('burst_peak_values', None)
    pre_burst_peak_times = plot_data.get('pre_burst_peak_times', None)
    pre_burst_peak_values = plot_data.get('pre_burst_peak_values', None)

    if len(t) == 0:
        return

    ax.plot(t, signal, color='#B22222', lw=1.5, zorder=2)
    if signal_smooth is not None and len(signal_smooth) > 0:
        ax.plot(t, signal_smooth, color='tab:orange', lw=1.0, zorder=3, alpha=0.7)
    if pre_burst_peak_times is not None and len(pre_burst_peak_times) > 0:
        ax.plot(pre_burst_peak_times, pre_burst_peak_values, 'o', color='steelblue',
                ms=5, zorder=4, alpha=0.8, label='Pre-burstlet peaks')
    if burst_peak_times is not None and len(burst_peak_times) > 0:
        ax.plot(burst_peak_times, burst_peak_values, 'o', color='red', ms=4, zorder=5,
                label='Burstlet peaks')
    if baseline is not None:
        ax.axhline(baseline, color='#FF6600', ls='--', lw=1, alpha=0.8, label='Baseline')
    if threshold is not None:
        ax.axhline(threshold, color='#C0392B', ls='--', lw=1, alpha=0.8, label='Burst thresh')
    if pre_burst_threshold is not None:
        ax.axhline(pre_burst_threshold, color='steelblue', ls='--', lw=1, alpha=0.8, label='Pre-burst thresh')

    ax.set_ylabel("Synchrony")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _overlay_bursts(ax, burst_events):
    """Overlay shaded regions for each hierarchical burst level."""
    colors = {
        'burstlets':      ('tab:green',  0.10),
        'network_bursts': ('tab:blue',   0.15),
        'superbursts':    ('purple',     0.20),
    }
    for level, (color, alpha) in colors.items():
        events = burst_events.get(level, [])
        for ev in events:
            ax.axvspan(ev['start'], ev['end'], color=color, alpha=alpha)


# =============================================================================
# MAIN FITNESS FUNCTION
# =============================================================================

def fitnessFunc_v2(
    simulated_data_path: str = None,
    reference_data_path: str = None,
    weights: Optional[Dict[str, float]] = None,
    config: Optional[Dict] = None,
    **kwargs,
) -> float:
    """
    Main fitness function for network optimization.

    Computes a multi-metric fitness score comparing simulated data to an
    experimental reference stored in experimental_features.h5.

    Returns
    -------
    fitness : float
        Scalar fitness value (lower is better).
        Also saves a detailed fitness.json alongside the simulation data.
    """
    time_start = time.time()
    # max_score = DEFAULT_CONFIG.get('max_fitness', 10000.0)
    max_score = 30000.0

    # --- weights ---
    if weights is None:
        try:
            weights = get_component_weights()
        except Exception:
            weights = {
                'unit_metrics': 0.30, 'quality_metrics': 0.0, 'synchrony': 0.10,
                'pre_burstlets': 0.15, 'burstlets': 0.20, 'network_bursts': 0.20, 'superbursts': 0.05,
            }
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    if config is None:
        config = DEFAULT_CONFIG.copy()

    simulated_data = None

    try:
        # ----------------------------------------------------------------
        # 1. Load simulated data
        # ----------------------------------------------------------------
        if kwargs.get('batching', False):
            kwargs = get_sim_data_from_call_stack(kwargs)
            simulated_data = kwargs.get('simData')
        elif kwargs.get('sim_data_path'):
            kwargs = get_sim_data_from_pkl(kwargs)
            simulated_data = kwargs.get('simData')
        else:
            # Direct dict or path
            if simulated_data_path is not None:
                if isinstance(simulated_data_path, str) and simulated_data_path.endswith('.pkl'):
                    if sim is not None:
                        sim.load(simulated_data_path)
                        simulated_data = sim.allSimData.todict().copy()
                        kwargs['simData'] = simulated_data
                    else:
                        simulated_data = np.load(simulated_data_path, allow_pickle=True)
                elif isinstance(simulated_data_path, dict):
                    simulated_data = simulated_data_path
                else:
                    simulated_data = np.load(simulated_data_path, allow_pickle=True)

        if simulated_data is None:
            logger.warning("No simulated data")
            return max_score

        # ----------------------------------------------------------------
        # 2. Load experimental data
        # ----------------------------------------------------------------
        if reference_data_path is None:
            reference_data_path = kwargs.get('reference_data_path')
        if reference_data_path is None:
            logger.error("No reference_data_path provided")
            return max_score

        logger.info(f"Loading experimental data from {reference_data_path}")
        exp_data = load_experimental_h5(reference_data_path)

        if exp_data['n_units'] == 0:
            logger.warning("No units in experimental data")
            return max_score

        # ----------------------------------------------------------------
        # 3. Extract simulated features
        # ----------------------------------------------------------------
        recording_duration = exp_data['recording_duration']
        network_cool_down = kwargs.get('network_cool_down', 0.0)
        min_analysis_time = network_cool_down if network_cool_down > 0 else None

        sim_features = extract_simulated_features(
            simulated_data, recording_duration, min_time=min_analysis_time,
        )
        if sim_features['n_units'] == 0:
            logger.warning("No spikes in simulation")
            return max_score

        # ----------------------------------------------------------------
        # 4. Compute scores
        # ----------------------------------------------------------------
        logger.info("Computing unit metrics")
        unit_results = compute_unit_score(sim_features, exp_data, recording_duration)

        logger.info("Computing synchrony metrics")
        sync_results = compute_synchrony_score(
            sim_features['spike_data'], exp_data,
            bin_size=config.get('synchrony', {}).get('bin_size', 0.01),
            recording_duration=recording_duration,
        )

        logger.info("Computing hierarchical burst metrics")
        burst_params = config.get('hierarchical_bursts', {})
        burst_params = {}
        # use_simple_burst_scoring = kwargs.get('use_simple_burst_scoring', False)
        use_simple_burst_scoring = False
        use_v2_burst_scoring = True
        # import pdb; pdb.set_trace()
        if use_v2_burst_scoring:
            logger.info("Using simple burst scoring v2 (count + timing MSE)")
            burst_level_weights = kwargs.get('burst_level_weights', None)
            burst_results = compute_hierarchical_burst_scores_simple_v2(
                sim_features['spike_data'], exp_data['spike_data'],
                recording_duration, burst_params,
                level_weights=burst_level_weights,
            )
        elif use_simple_burst_scoring:
            logger.info("Using simple burst scoring (event count MSE)")
            burst_results = compute_hierarchical_burst_scores_simple(
                sim_features['spike_data'], exp_data['spike_data'],
                recording_duration, burst_params,
            )
        else:
            logger.info("Using full burst scoring (weighted multi-metric)")
            burst_results = compute_hierarchical_burst_scores(
                sim_features['spike_data'], exp_data['spike_data'],
                recording_duration, burst_params,
            )

        burstlet_score      = burst_results['burstlets']['total_score']
        network_burst_score = burst_results['network_bursts']['total_score']
        superburst_score    = burst_results['superbursts']['total_score']
        pre_burstlet_score  = burst_results.get('pre_burstlets', {}).get('total_score', 0.0)

        unit_score = unit_results['total_score']
        sync_score = sync_results['total_score']

        # Clamp
        def _clamp(v):
            return min(v, max_score) if np.isfinite(v) else max_score

        unit_score          = _clamp(unit_score)
        sync_score          = _clamp(sync_score)
        pre_burstlet_score  = _clamp(pre_burstlet_score)
        burstlet_score      = _clamp(burstlet_score)
        network_burst_score = _clamp(network_burst_score)
        superburst_score    = _clamp(superburst_score)

        fitness = (
            weights.get('unit_metrics',    0.0) * unit_score +
            weights.get('synchrony',       0.0) * sync_score +
            weights.get('pre_burstlets',   0.0) * pre_burstlet_score +
            weights.get('burstlets',       0.0) * burstlet_score +
            weights.get('network_bursts',  0.0) * network_burst_score +
            weights.get('superbursts',     0.0) * superburst_score
        )
        fitness = min(fitness, max_score)

        # ----------------------------------------------------------------
        # 5. Build result dict
        # ----------------------------------------------------------------
        result = {
            'fitness': float(fitness),
            'fit': float(fitness),
            'fitness_dict': {
                'unit_metrics': {'fit': float(unit_score), **unit_results},
                'synchrony': {'fit': float(sync_score), **sync_results},
                'pre_burstlets': {'fit': float(pre_burstlet_score), **burst_results.get('pre_burstlets', {})},
                'burstlets': {'fit': float(burstlet_score), **burst_results['burstlets']},
                'network_bursts': {'fit': float(network_burst_score), **burst_results['network_bursts']},
                'superbursts': {'fit': float(superburst_score), **burst_results['superbursts']},
                'hierarchical_diagnostics': burst_results['diagnostics'],
                'fit': float(fitness),
            },
            'simulated_metrics': {
                'n_units': sim_features['n_units'],
                'total_spikes': sim_features['total_spikes'],
                'mean_firing_rate': sim_features['mean_firing_rate'],
                'recording_duration': recording_duration,
            },
            'experimental_metrics': {
                'n_units': exp_data['n_units'],
                'total_spikes': exp_data['total_spikes'],
                'mean_firing_rate': exp_data['mean_firing_rate'],
                'recording_duration': exp_data['recording_duration'],
            },
            'weights': weights,
        }

        logger.info(f"Fitness = {fitness:.4f}")

        # ----------------------------------------------------------------
        # 6. Save fitness.json
        # ----------------------------------------------------------------
        candidate_path = kwargs.get('candidate_path')
        sim_data_path_kw = kwargs.get('sim_data_path')
        if candidate_path is None and sim_data_path_kw and isinstance(sim_data_path_kw, str):
            candidate_path = sim_data_path_kw.replace('_data.pkl', '').replace('.npy', '')

        fitness_save_path = kwargs.get('fitness_save_path')
        if fitness_save_path is None and candidate_path:
            fitness_save_path = f'{candidate_path}_fitness.json'

        if fitness_save_path:
            # Make JSON-safe
            def _json_safe(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, dict):
                    return {str(k): _json_safe(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_json_safe(v) for v in obj]
                return obj

            with open(fitness_save_path, 'w') as fp:
                json.dump(_json_safe(result), fp, indent=2, default=str)
            logger.info(f"Saved fitness to {fitness_save_path}")

        # ----------------------------------------------------------------
        # 7. Plot
        # ----------------------------------------------------------------
        plot_enabled = kwargs.get('plot_sim', False)
        if plot_enabled and candidate_path:
            plot_path = f'{candidate_path}_fitness_plot.png'
            try:
                plot_fitness_comparison(
                    sim_features, exp_data, burst_results,
                    save_path=plot_path,
                    title_prefix=f"Fitness={fitness:.2f} | ",
                )
            except Exception as e:
                logger.warning(f"Plotting failed: {e}")

        elapsed = time.time() - time_start
        logger.info(f"Elapsed: {elapsed:.2f}s")

        return float(fitness)

    except Exception as e:
        logger.error(f"Error computing fitness: {e}")
        traceback.print_exc()
        return max_score


# =============================================================================
# CONVENIENCE / STANDALONE
# =============================================================================

def compute_fitness(
    sim_data: Any,
    exp_h5_path: str,
    weights: Optional[Dict] = None,
) -> Tuple[float, Dict]:
    """Convenience wrapper returning (fitness_value, result_dict)."""
    fitness = fitnessFunc_v2(
        simulated_data_path=sim_data,
        reference_data_path=exp_h5_path,
        weights=weights,
    )
    return fitness, {}


def fitness_report(result: Dict) -> str:
    """Pretty-print a fitness result dict."""
    lines = ["=" * 60, "FITNESS REPORT", "=" * 60]
    lines.append(f"  Overall fitness:  {result.get('fitness', 'N/A'):.4f}")
    lines.append("")
    fd = result.get('fitness_dict', {})
    for comp_name in ['unit_metrics', 'synchrony', 'burstlets', 'network_bursts', 'superbursts']:
        comp = fd.get(comp_name, {})
        lines.append(f"  {comp_name:20s}  score = {comp.get('fit', comp.get('total_score', 'N/A'))}")
    lines.append("")
    sm = result.get('simulated_metrics', {})
    em = result.get('experimental_metrics', {})
    lines.append(f"  Sim units: {sm.get('n_units', '?')}   Exp units: {em.get('n_units', '?')}")
    lines.append(f"  Sim spikes: {sm.get('total_spikes', '?')}  Exp spikes: {em.get('total_spikes', '?')}")
    lines.append("=" * 60)
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fitness Function v2 — standalone mode")
    parser.add_argument("--sim", type=str, required=True,
                        help="Path to simulated data (.pkl or .npy)")
    parser.add_argument("--ref", type=str, required=True,
                        help="Path to experimental_features.h5")
    parser.add_argument("--plot", action="store_true",
                        help="Generate comparison plots")
    parser.add_argument("--output", type=str, default=None,
                        help="Output fitness.json path (default: <sim>_fitness.json)")

    args = parser.parse_args()

    candidate_path = args.sim.replace('_data.pkl', '').replace('.npy', '')
    fitness_save = args.output or f"{candidate_path}_fitness.json"

    fitness = fitnessFunc_v2(
        simulated_data_path=args.sim,
        reference_data_path=args.ref,
        plot_sim=args.plot,
        candidate_path=candidate_path,
        fitness_save_path=fitness_save,
    )

    print(f"\nFitness: {fitness:.4f}")
    print(f"Saved:   {fitness_save}")
