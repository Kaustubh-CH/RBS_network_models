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
import pickle
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
        fit_schema as DEFAULT_FIT_SCHEMA,
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
            fit_schema as DEFAULT_FIT_SCHEMA,
            normalize_schema_weights,
        )
    except ImportError:
        # Minimal fallback defaults
        DEFAULT_CONFIG = {'max_fitness': 1000.0, 'recording_duration': 300.0}
        DEFAULT_FIT_SCHEMA = {}
        normalize_schema_weights = lambda: DEFAULT_FIT_SCHEMA

# matplotlib — imported lazily in plot function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _resolve_fit_schema(schema_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Use the schema passed in kwargs when present, otherwise fall back to the module default."""
    if isinstance(schema_override, dict):
        return schema_override
    return DEFAULT_FIT_SCHEMA


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

def _load_drug_simData_from_pkl(pkl_path: str) -> Dict:
    """Read drug_simData (if any) directly from the pkl on disk.

    NetPyNE's ``sim.load(pkl_path)`` unpacks only the keys it knows about
    (simData, simConfig, netParams, etc.) into globals — our extra
    ``drug_simData`` key, merged in by init.py for multi-drug runs, is
    silently dropped. This helper re-opens the pkl with pickle to pull it out.
    Returns ``{}`` when the key is absent (legacy single-condition pkl) or
    the pkl cannot be read.
    """
    try:
        with open(pkl_path, 'rb') as _f:
            _pkl = pickle.load(_f)
        if isinstance(_pkl, dict):
            return _pkl.get('drug_simData', {}) or {}
    except Exception as _e:
        logger.warning(f"Could not read drug_simData from {pkl_path}: {_e}")
    return {}


def get_sim_data_from_call_stack(kwargs: Dict) -> Dict:
    """Load simulated data from NetPyNE batch call stack."""
    if sim is None:
        raise ImportError("netpyne not available")
    from .utils.utils_old.extract_simulated_data import get_candidate_and_job_path_from_call_stack
    # from .utils.utils_old.batch_helper import get_candidate_and_job_path_from_call_stack
    candidate_path, candidate_label = get_candidate_and_job_path_from_call_stack()
    pkl_path = f'{candidate_path}_data.pkl'
    sim.load(pkl_path)

    sim_cfg_dict = sim.cfg.todict().copy()
    if hasattr(sim.cfg, 'excit_units'):
        sim_cfg_dict['excit_units'] = list(getattr(sim.cfg, 'excit_units'))
    if hasattr(sim.cfg, 'inhib_units'):
        sim_cfg_dict['inhib_units'] = list(getattr(sim.cfg, 'inhib_units'))

    kwargs.update({
        'simData': sim.allSimData.todict().copy(),
        'cellData': sim.net.allCells.copy(),
        'popData': sim.net.allPops.copy(),
        'simCfg': sim_cfg_dict,
        'netParams': sim.net.params.todict().copy(),
        'simLabel': candidate_label,
        'data_file_path': pkl_path,
        'candidate_path': candidate_path,
        'fitness_save_path': f'{candidate_path}_fitness.json',
        'metrics_save_path': f'{candidate_path}_metrics.npy',
        'drug_simData': _load_drug_simData_from_pkl(pkl_path),
    })
    return kwargs


def get_sim_data_from_pkl(kwargs: Dict) -> Dict:
    """Load simulated data from a .pkl path stored in kwargs['sim_data_path']."""
    if sim is None:
        raise ImportError("netpyne not available")
    sim_data_path = kwargs['sim_data_path']
    sim.load(sim_data_path)
    candidate_path = sim_data_path.replace('_data.pkl', '')

    sim_cfg_dict = sim.cfg.todict().copy()
    if hasattr(sim.cfg, 'excit_units'):
        sim_cfg_dict['excit_units'] = list(getattr(sim.cfg, 'excit_units'))
    if hasattr(sim.cfg, 'inhib_units'):
        sim_cfg_dict['inhib_units'] = list(getattr(sim.cfg, 'inhib_units'))

    kwargs.update({
        'simData': sim.allSimData.todict().copy(),
        'cellData': sim.net.allCells.copy(),
        'popData': sim.net.allPops.copy(),
        'simCfg': sim_cfg_dict,
        'netParams': sim.net.params.todict().copy(),
        'simLabel': os.path.basename(candidate_path),
        'data_file_path': sim_data_path,
        'candidate_path': candidate_path,
        'fitness_save_path': f'{candidate_path}_fitness.json',
        'metrics_save_path': f'{candidate_path}_metrics.npy',
        'drug_simData': _load_drug_simData_from_pkl(sim_data_path),
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


def compute_unit_score(
    sim_features: Dict,
    exp_data: Dict,
    recording_duration: float,
    sim_excit_units: Optional[List[int]] = None,
    sim_inhib_units: Optional[List[int]] = None,
    fit_schema: Optional[Dict[str, Any]] = None,
    pop_data: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Compare per-unit metrics between sim and experiment."""
    max_score = DEFAULT_CONFIG.get('max_fitness', 1000.0)
    fit_schema = _resolve_fit_schema(fit_schema)

    # -- firing rates (overall) --
    sim_frs = list(sim_features['firing_rates'].values())
    exp_frs = [
        exp_data['unit_attrs'][uid].get('firing_rate', 0.0)
        for uid in exp_data['unit_ids']
    ]
    sim_mean_fr, _ = _safe_stat(sim_frs)
    exp_mean_fr, _ = _safe_stat(exp_frs)
    firing_rate_error_all = ((sim_mean_fr - exp_mean_fr) ** 2) / max(exp_mean_fr ** 2, 1e-6) * 100.0

    # -- firing rates split by class (E/I) --
    exp_exc_uids = []
    exp_inh_uids = []
    for uid in exp_data['unit_ids']:
        ct = str(exp_data['unit_attrs'][uid].get('cell_type', 'excitatory')).strip().lower()
        if ct == 'inhibitory':
            exp_inh_uids.append(uid)
        else:
            exp_exc_uids.append(uid)

    exp_exc_frs = [exp_data['unit_attrs'][uid].get('firing_rate', 0.0) for uid in exp_exc_uids]
    exp_inh_frs = [exp_data['unit_attrs'][uid].get('firing_rate', 0.0) for uid in exp_inh_uids]
    exp_mean_fr_exc, _ = _safe_stat(exp_exc_frs)
    exp_mean_fr_inh, _ = _safe_stat(exp_inh_frs)

    sim_fr_map = sim_features.get('firing_rates', {})

    # --- Build sim E/I GID sets from popData (GID-based) ---
    sim_exc_uids: set = set()
    sim_inh_uids: set = set()
    if pop_data:
        for pop_name, pop_info in pop_data.items():
            cell_gids = pop_info.get('cellGids', [])
            if pop_name == 'I':
                sim_inh_uids.update(int(gid) for gid in cell_gids)
            else:
                sim_exc_uids.update(int(gid) for gid in cell_gids)
        logger.info(f"compute_unit_score: Built sim E/I from popData: "
                    f"{len(sim_exc_uids)} E, {len(sim_inh_uids)} I")
    else:
        # Fallback: use counts from excit/inhib unit lists
        # E population gets GIDs 0..num_excite-1, I gets the rest
        num_e = len(sim_excit_units or [])
        num_i = len(sim_inhib_units or [])
        if num_e > 0 or num_i > 0:
            sim_exc_uids = set(range(num_e))
            sim_inh_uids = set(range(num_e, num_e + num_i))
            logger.info(f"compute_unit_score: Built sim E/I from unit counts: "
                        f"{num_e} E, {num_i} I")
        else:
            logger.warning("compute_unit_score: No E/I labels available; all units treated as excitatory.")
            sim_exc_uids = set(int(uid) for uid in sim_fr_map.keys())

    sim_exc_frs = [float(sim_fr_map[uid]) for uid in sim_fr_map if int(uid) in sim_exc_uids]
    sim_inh_frs = [float(sim_fr_map[uid]) for uid in sim_fr_map if int(uid) in sim_inh_uids]
    sim_mean_fr_exc, _ = _safe_stat(sim_exc_frs)
    sim_mean_fr_inh, _ = _safe_stat(sim_inh_frs)

    firing_rate_error_exc = ((sim_mean_fr_exc - exp_mean_fr_exc) ** 2) / max(exp_mean_fr_exc ** 2, 1e-6) * 100.0
    firing_rate_error_inh = ((sim_mean_fr_inh - exp_mean_fr_inh) ** 2) / max(exp_mean_fr_inh ** 2, 1e-6) * 100.0

    n_exp_exc = len(exp_exc_uids)
    n_exp_inh = len(exp_inh_uids)

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
        str(exp_data['unit_attrs'][uid].get('cell_type', 'excitatory')).strip().lower()
        for uid in exp_data['unit_ids']
    ]
    exp_inh_frac = sum(1 for t in exp_types if t == 'inhibitory') / max(len(exp_types), 1)
    if sim_fr_map:
        sim_inh_frac = len(sim_inh_uids) / max(len(sim_fr_map), 1)
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

    fr_w_exc = float(_w('firing_rate_error_exc')) if 'firing_rate_error_exc' in schema else 0.0
    fr_w_inh = float(_w('firing_rate_error_inh')) if 'firing_rate_error_inh' in schema else 0.0
    if (fr_w_exc + fr_w_inh) > 0:
        norm = fr_w_exc + fr_w_inh
        fr_w_exc /= norm
        fr_w_inh /= norm
    else:
        n_exp_total = max(n_exp_exc + n_exp_inh, 1)
        fr_w_exc = n_exp_exc / n_exp_total
        fr_w_inh = n_exp_inh / n_exp_total

    firing_rate_error = fr_w_exc * firing_rate_error_exc + fr_w_inh * firing_rate_error_inh

    total = (
        _w('firing_rate_error') * _clip(firing_rate_error, 'firing_rate_error') +
        _w('firing_rate_error_exc') * _clip(firing_rate_error_exc, 'firing_rate_error_exc') +
        _w('firing_rate_error_inh') * _clip(firing_rate_error_inh, 'firing_rate_error_inh') +
        _w('num_spikes_error') * _clip(num_spikes_error, 'num_spikes_error') +
        _w('firing_range_error') * _clip(firing_range_error, 'firing_range_error') +
        _w('cv_isi_error') * _clip(cv_isi_error, 'cv_isi_error') +
        _w('n_units_error') * _clip(n_units_error, 'n_units_error') +
        _w('ei_ratio_error') * _clip(ei_ratio_error, 'ei_ratio_error')
    )

    return {
        'firing_rate_error': float(firing_rate_error),
        'firing_rate_error_all': float(firing_rate_error_all),
        'firing_rate_error_exc': float(firing_rate_error_exc),
        'firing_rate_error_inh': float(firing_rate_error_inh),
        'sim_mean_firing_rate': float(sim_mean_fr),
        'exp_mean_firing_rate': float(exp_mean_fr),
        'sim_mean_firing_rate_exc': float(sim_mean_fr_exc),
        'exp_mean_firing_rate_exc': float(exp_mean_fr_exc),
        'sim_mean_firing_rate_inh': float(sim_mean_fr_inh),
        'exp_mean_firing_rate_inh': float(exp_mean_fr_inh),
        'sim_n_exc': int(len(sim_exc_frs)),
        'sim_n_inh': int(len(sim_inh_frs)),
        'exp_n_exc': int(n_exp_exc),
        'exp_n_inh': int(n_exp_inh),
        'firing_rate_weight_exc': float(fr_w_exc),
        'firing_rate_weight_inh': float(fr_w_inh),
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
    fit_schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compare synchrony metrics."""
    fit_schema = _resolve_fit_schema(fit_schema)
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
    fit_schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Score one hierarchical burst level (burstlets / network_bursts / superbursts).

    Every metric is normalised by its experimental value and expressed as a
    squared relative error::

        metric_error = ((sim_val - exp_val) / max(|exp_val|, 1e-6)) ** 2

    The total score is a schema-weighted combination of per-metric errors,
    scaled by 100 for readability.

    Includes a Victor-Purpura timing error based on event start times.
    """
    _EPS = 1e-6
    fit_schema = _resolve_fit_schema(fit_schema)

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

    # -- timing (Victor-Purpura distance on event starts) --
    sim_starts = np.array(sorted(ev.get('start', 0.0) for ev in sim_events), dtype=float)
    exp_starts = np.array(sorted(ev.get('start', 0.0) for ev in exp_events), dtype=float)
    timing_error = _victor_purpura_distance(sim_starts, exp_starts, q=1.0) / max(n_exp, 1)

    all_errors = {
        'burst_count_error': count_error,
        'burst_rate_error': rate_error,
        'duration_error': duration_error,
        'ibi_error': ibi_error,
        'participation_error': participation_error,
        'intensity_error': intensity_error,
        'spikes_per_burst_error': spikes_per_burst_error,
        'timing_error': timing_error,
    }

    # -- schema-weighted total (fallback: unweighted mean) --
    schema = fit_schema.get(level_name, {}).get('metrics', {})
    weighted_sum = 0.0
    weight_total = 0.0
    for metric_name, metric_cfg in schema.items():
        if metric_name not in all_errors:
            continue
        w = float(metric_cfg.get('weight', 0.0))
        max_val = float(metric_cfg.get('max_val', DEFAULT_CONFIG.get('max_fitness', 1000.0)))
        val = min(float(all_errors[metric_name]), max_val)
        weighted_sum += w * val
        weight_total += w

    if weight_total > 0:
        total = (weighted_sum / weight_total) * 100.0
    else:
        total = float(np.mean(list(all_errors.values()))) * 100.0

    return {
        'burst_count_error': float(count_error),
        'burst_rate_error': float(rate_error),
        'duration_error': float(duration_error),
        'ibi_error': float(ibi_error),
        'participation_error': float(participation_error),
        'intensity_error': float(intensity_error),
        'spikes_per_burst_error': float(spikes_per_burst_error),
        'timing_error': float(timing_error),
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
        'sim_timing_starts_count': int(len(sim_starts)),
        'exp_timing_starts_count': int(len(exp_starts)),
        'total_score': float(total),
    }


def compute_hierarchical_burst_scores(
    sim_spike_data: Dict[int, np.ndarray],
    exp_spike_data: Dict[int, np.ndarray],
    recording_duration: float,
    burst_params: Optional[Dict] = None,
    fit_schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run parameter_free_burst_detector on both sim and exp, then score
    each hierarchical level.

    Uses base_threshold_multiplier=45 and min_burstlet_participation=0.05
    for the main burst detection, and base_threshold_multiplier=15 with
    min_burstlet_participation=0.05 for pre-burstlet detection.
    """
    bp = burst_params or {}
    fit_schema = _resolve_fit_schema(fit_schema)

    # Main burst detection (static threshold = 40 on ws_sharp)
    sim_bursts = _run_burst_detector(
        sim_spike_data, min_burstlet_participation=0.05,
        base_threshold_static=40, **bp,
    )
    exp_bursts = _run_burst_detector(
        exp_spike_data, min_burstlet_participation=0.05,
        base_threshold_static=40, **bp,
    )

    results = {}
    for level in ['burstlets', 'network_bursts', 'superbursts']:
        sim_evts = sim_bursts.get(level, {}).get('events', [])
        exp_evts = exp_bursts.get(level, {}).get('events', [])
        sim_met  = sim_bursts.get(level, {}).get('metrics', {})
        exp_met  = exp_bursts.get(level, {}).get('metrics', {})
        results[level] = _score_burst_level(
            sim_evts, exp_evts, sim_met, exp_met, recording_duration, level,
            fit_schema=fit_schema,
        )

    # ------------------------------------------------------------------
    # Pre-burstlet score (lower threshold = 15×)
    # ------------------------------------------------------------------
    sim_pre_bursts = _run_burst_detector(
        sim_spike_data, base_threshold_static=15,
        min_burstlet_participation=0.05,
    )
    exp_pre_bursts = _run_burst_detector(
        exp_spike_data, base_threshold_static=15,
        min_burstlet_participation=0.05,
    )

    sim_pre_evts = sim_pre_bursts.get('burstlets', {}).get('events', [])
    exp_pre_evts = exp_pre_bursts.get('burstlets', {}).get('events', [])
    sim_pre_met  = sim_pre_bursts.get('burstlets', {}).get('metrics', {})
    exp_pre_met  = exp_pre_bursts.get('burstlets', {}).get('metrics', {})

    results['pre_burstlets'] = _score_burst_level(
        sim_pre_evts, exp_pre_evts, sim_pre_met, exp_pre_met,
        recording_duration, 'pre_burstlets',
        fit_schema=fit_schema,
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
    fit_schema: Optional[Dict[str, Any]] = None,
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
    fit_schema = _resolve_fit_schema(fit_schema)

    sim_bursts = _run_burst_detector(sim_spike_data, min_burstlet_participation=0.05, base_threshold_static=50, **bp)
    exp_bursts = _run_burst_detector(exp_spike_data, base_threshold_static=45, **bp)

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
                'peak_width_score': _max_score,
                'amp_score': _max_score,
                'total_score': _max_score,
                'weights': {'count': 0.4, 'timing': 0.2, 'peak_width': 0.4, 'amp': 0.0},
            }
            for level in _levels
        }
        _r['pre_burstlets'] = {
            'n_sim': 0, 'n_exp': 0,
            'count_score': _max_score, 'timing_score': _max_score, 'peak_width_score': _max_score,
            'amp_score': _max_score,
            'total_score': _max_score,
            'weights': {'count': 0.4, 'timing': 0.2, 'peak_width': 0.4, 'amp': 0.0},
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

    default_w = {'count': 0.4, 'timing': 0.2, 'peak_width': 0.4, 'amp': 0.0}

    # Burst amplitude in units of peak population rate PER UNIT. The detector's
    # PFR is a raw sum of per-unit spike counts (it is never divided by unit
    # count), so a raw burst_peak scales with network size -- comparing sim to exp
    # without this division would mostly measure the unit-count mismatch between
    # the model and the recording rather than anything about the bursts.
    _n_sim_units = max(len(sim_spike_data), 1)
    _n_exp_units = max(len(exp_spike_data), 1)

    def _mean_amp(events, n_units: int) -> float:
        if not events:
            return 0.0
        return float(np.mean([float(ev.get('burst_peak', 0.0)) for ev in events])) / n_units

    def _weights_from_schema_dict(schema_dict: Dict[str, Any], level_name: str) -> Dict[str, float]:
        # schema_v2-style structure: fit_schema[level]['metrics'][metric]['weight']
        # if pre_burstlets missing, reuse burstlets metric weights
        level_block = schema_dict.get(level_name)
        if level_block is None and level_name == 'pre_burstlets':
            level_block = schema_dict.get('burstlets')

        metrics = level_block.get('metrics', {}) if isinstance(level_block, dict) else {}
        return {
            'count': float(metrics.get('burst_count_error', {}).get('weight', 0.0)),
            'timing': float(metrics.get('timing_error', {}).get('weight', 0.0)),
            # Requested mapping: duration_error weight drives spike-width term
            'peak_width': float(metrics.get('duration_error', {}).get('weight', 0.0)),
            # Schemas without burst_amp_error resolve to 0.0 here, so the amplitude
            # term is inert and older schemas score identically to before.
            'amp': float(metrics.get('burst_amp_error', {}).get('weight', 0.0)),
        }

    def _resolve_weights(level_name: str) -> Dict[str, float]:
        schema_w = _weights_from_schema_dict(fit_schema, level_name)

        if float(sum(schema_w.values())) <= 0:
            schema_w = default_w.copy()

        w = schema_w
        s = float(w.get('count', 0.0) + w.get('timing', 0.0)
                  + w.get('peak_width', 0.0) + w.get('amp', 0.0))
        if s <= 0:
            return schema_w
        return {
            'count': float(w.get('count', 0.0)) / s,
            'timing': float(w.get('timing', 0.0)) / s,
            'peak_width': float(w.get('peak_width', 0.0)) / s,
            'amp': float(w.get('amp', 0.0)) / s,
        }

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

        # --- Peak width score: relative error of mean peak width ---
        sim_peak_widths = [max(0.0, float(ev.get('peak_width_s', 0.0))) for ev in sim_events]
        exp_peak_widths = [max(0.0, float(ev.get('peak_width_s', 0.0))) for ev in exp_events]
        sim_mean_peak_width = float(np.mean(sim_peak_widths)) if sim_peak_widths else 0.0
        exp_mean_peak_width = float(np.mean(exp_peak_widths)) if exp_peak_widths else 0.0
        peak_width_se = abs(sim_mean_peak_width - exp_mean_peak_width) / max(exp_mean_peak_width, 1e-6)

        # --- Amplitude score: relative error of mean per-unit burst peak ---
        sim_mean_amp = _mean_amp(sim_events, _n_sim_units)
        exp_mean_amp = _mean_amp(exp_events, _n_exp_units)
        amp_se = abs(sim_mean_amp - exp_mean_amp) / max(exp_mean_amp, 1e-6)

        # --- Combined score for this level ---
        w = _resolve_weights(level)
        w_count = w['count']
        w_timing = w['timing']
        w_peak_width = w['peak_width']
        w_amp = w['amp']
        level_score = (w_count * count_se + w_timing * timing_vp
                       + w_peak_width * peak_width_se + w_amp * amp_se) * 100

        results[level] = {
            'n_sim': n_sim,
            'n_exp': n_exp,
            'count_score': count_se,
            'timing_score': timing_vp,
            'peak_width_score': peak_width_se,
            'amp_score': amp_se,
            'sim_mean_peak_width': sim_mean_peak_width,
            'exp_mean_peak_width': exp_mean_peak_width,
            'sim_mean_amp': sim_mean_amp,
            'exp_mean_amp': exp_mean_amp,
            'total_score': level_score,
            'weights': {'count': w_count, 'timing': w_timing,
                        'peak_width': w_peak_width, 'amp': w_amp},
        }
        level_scores.append(level_score)

    mse = float(np.mean(level_scores))
    results['mse'] = mse
    results['total_score'] = mse

    # ------------------------------------------------------------------
    # Pre-burstlet score (lower threshold = 1.2×)
    # ------------------------------------------------------------------
    # pre_bp = {**bp, 'base_threshold_multiplier': 1.2}
    sim_pre_bursts = _run_burst_detector(sim_spike_data, base_threshold_static=30, min_burstlet_participation=0.05)
    exp_pre_bursts = _run_burst_detector(exp_spike_data, base_threshold_static=30, min_burstlet_participation=0.05)

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

    sim_pre_peak_widths = [max(0.0, float(ev.get('peak_width_s', 0.0))) for ev in sim_pre_events]
    exp_pre_peak_widths = [max(0.0, float(ev.get('peak_width_s', 0.0))) for ev in exp_pre_events]
    sim_pre_mean_peak_width = float(np.mean(sim_pre_peak_widths)) if sim_pre_peak_widths else 0.0
    exp_pre_mean_peak_width = float(np.mean(exp_pre_peak_widths)) if exp_pre_peak_widths else 0.0
    pre_peak_width_se = abs(sim_pre_mean_peak_width - exp_pre_mean_peak_width) / max(exp_pre_mean_peak_width, 1e-6)

    sim_pre_mean_amp = _mean_amp(sim_pre_events, _n_sim_units)
    exp_pre_mean_amp = _mean_amp(exp_pre_events, _n_exp_units)
    pre_amp_se = abs(sim_pre_mean_amp - exp_pre_mean_amp) / max(exp_pre_mean_amp, 1e-6)

    pre_w = _resolve_weights('pre_burstlets')
    pre_score = (pre_w['count'] * pre_count_se + pre_w['timing'] * pre_timing_vp
                 + pre_w['peak_width'] * pre_peak_width_se + pre_w['amp'] * pre_amp_se) * 100

    sim_pre_threshold = sim_pre_bursts.get('plot_data', {}).get('threshold', None)
    exp_pre_threshold = exp_pre_bursts.get('plot_data', {}).get('threshold', None)

    results['pre_burstlets'] = {
        'n_sim': n_sim_pre,
        'n_exp': n_exp_pre,
        'count_score': pre_count_se,
        'timing_score': pre_timing_vp,
        'peak_width_score': pre_peak_width_se,
        'amp_score': pre_amp_se,
        'sim_mean_peak_width': sim_pre_mean_peak_width,
        'exp_mean_peak_width': exp_pre_mean_peak_width,
        'sim_mean_amp': sim_pre_mean_amp,
        'exp_mean_amp': exp_pre_mean_amp,
        'total_score': pre_score,
        'weights': {'count': pre_w['count'], 'timing': pre_w['timing'],
                    'peak_width': pre_w['peak_width'], 'amp': pre_w['amp']},
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
    sim_excit_units: Optional[List[int]] = None,
    sim_inhib_units: Optional[List[int]] = None,
    pop_data: Optional[Dict] = None,
) -> None:
    """
    Generate a multi-panel plot:
        Row 1: Simulated raster
        Row 2: Simulated network activity + burst overlays
        Row 3: Experimental raster
        Row 4: Experimental network activity + burst overlays
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    ax_sim_raster, ax_sim_net, ax_exp_raster, ax_exp_net = axes

    # --- Build cell_types for simulated data from popData GID ranges ---
    sim_cell_types = {}
    if pop_data:
        for pop_name, pop_info in pop_data.items():
            cell_gids = pop_info.get('cellGids', [])
            ct = 'inhibitory' if pop_name == 'I' else 'excitatory'
            for gid in cell_gids:
                sim_cell_types[int(gid)] = ct
        logger.info(f"Built sim_cell_types from popData: "
                    f"{sum(1 for v in sim_cell_types.values() if v == 'excitatory')} E, "
                    f"{sum(1 for v in sim_cell_types.values() if v == 'inhibitory')} I")
    else:
        # Fallback: E population gets GIDs 0..num_excite-1, I gets the rest
        num_e = len(sim_excit_units or [])
        num_i = len(sim_inhib_units or [])
        for gid in range(num_e):
            sim_cell_types[gid] = 'excitatory'
        for gid in range(num_e, num_e + num_i):
            sim_cell_types[gid] = 'inhibitory'
        logger.info(f"Built sim_cell_types from unit counts: {num_e} E, {num_i} I")

    # --- Build cell_types for experimental data ---
    exp_cell_types = {}
    for uid in exp_data.get('unit_ids', []):
        ct = exp_data.get('unit_attrs', {}).get(uid, {}).get('cell_type', 'excitatory')
        exp_cell_types[uid] = ct
    
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
    """Clean raster plot — one row per unit, color-coded by cell type.
    Units are sorted by cell type: excitatory first (y=0..numExc-1),
    then inhibitory (y=numExc..), with a dashed separator line."""
    if cell_types is not None:
        exc_units = sorted(uid for uid in spike_data.keys()
                           if cell_types.get(uid, 'excitatory') != 'inhibitory')
        inh_units = sorted(uid for uid in spike_data.keys()
                           if cell_types.get(uid, 'excitatory') == 'inhibitory')
        units = exc_units + inh_units
        n_exc = len(exc_units)
    else:
        units = sorted(spike_data.keys())
        n_exc = None

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
            linestyle='None', marker='|', markersize=5,
            markeredgewidth=1.0, color=color, alpha=0.8, rasterized=True,
        )
    ax.set_ylabel("Unit Index")
    ax.set_ylim(-1, len(units))
    ax.set_title(title, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Draw separator line between E and I populations
    if n_exc is not None and 0 < n_exc < len(units):
        ax.axhline(y=n_exc - 0.5, color='black', linestyle='--',
                   linewidth=0.8, alpha=0.6)

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


def _sim_cell_types_from_popdata(pop_data, baseline_features=None) -> Dict[int, str]:
    """Map GID -> 'excitatory'/'inhibitory' from NetPyNE popData GID ranges.
    Mirrors the block inside plot_fitness_comparison so the drug rasters share
    the same E/I colouring as the baseline fitness plot."""
    sim_cell_types: Dict[int, str] = {}
    if pop_data:
        for pop_name, pop_info in pop_data.items():
            ct = 'inhibitory' if pop_name == 'I' else 'excitatory'
            for gid in pop_info.get('cellGids', []):
                sim_cell_types[int(gid)] = ct
    return sim_cell_types


def plot_drug_response(
    drugs: List[str],
    drug_simData: Dict[str, Dict],
    baseline_features: Dict,
    drug_results: Dict[str, Any],
    recording_duration: float,
    save_path_prefix: str,
    burst_params: Optional[Dict] = None,
    baseline_burst_results: Optional[Dict] = None,
    pop_data: Optional[Dict] = None,
    network_cool_down: float = 0.0,
) -> None:
    """One figure per drug comparing the perturbed network to baseline.

    Layout (rows, sharing the time axis for the first four):
        1. Baseline raster        (+ burst overlays)
        2. Baseline network activity
        3. Drug raster            (+ burst overlays)
        4. Drug network activity
        5. Bar chart: simulated vs experimental post/pre ratios per feature,
           annotated with the per-feature |sim-exp| deviation.

    Saves ``<prefix>_drug_<drug>.png`` (full) plus a 30s zoom of the rasters.
    Drugs that failed or are missing from ``drug_simData`` get a single
    annotated panel instead of the full comparison.
    """
    import matplotlib.pyplot as plt

    bp = burst_params or {}
    sim_cell_types = _sim_cell_types_from_popdata(pop_data, baseline_features)
    per_drug = (drug_results or {}).get('per_drug', {})

    # Baseline burst plot data / events: reuse what the baseline scoring already
    # computed when available; otherwise run the detector once here.
    if baseline_burst_results and baseline_burst_results.get('sim_plot_data'):
        base_pd = baseline_burst_results.get('sim_plot_data', {})
        base_events = baseline_burst_results.get('sim_burst_events', {})
    else:
        _bnb = _run_burst_detector(baseline_features.get('spike_data', {}),
                                   min_burstlet_participation=0.05, **bp)
        base_pd = _bnb.get('plot_data', {})
        base_events = {lvl: _bnb.get(lvl, {}).get('events', [])
                       for lvl in ['burstlets', 'network_bursts', 'superbursts']}

    for drug in drugs:
        entry = (drug_simData or {}).get(drug)
        drug_info = per_drug.get(drug, {})
        drug_fit = drug_info.get('fit')
        save_path = f"{save_path_prefix}_drug_{drug}.png"

        # --- Failed / missing drug: single annotated panel ---
        if not entry or entry.get('_failure') or 'sim_ratios' not in drug_info:
            reason = drug_info.get('reason') or (
                entry.get('_failure_reason') if isinstance(entry, dict) else 'missing')
            fig, ax = plt.subplots(1, 1, figsize=(10, 3))
            ax.axis('off')
            ax.text(0.5, 0.5,
                    f"Drug '{drug}' not scored\nreason: {reason}\n"
                    f"fit = {drug_fit}",
                    ha='center', va='center', fontsize=12)
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            logger.info(f"Saved drug placeholder plot: {save_path}")
            continue

        # --- Drug-side features + burst detection (first network_cool_down s
        # excluded + shifted, mirroring the baseline path so the rasters /
        # network-activity panels show only the settled analysis window) ---
        drug_features = _build_features_from_drug_pkl(entry, recording_duration,
                                                      min_time=network_cool_down)
        drug_nb = _run_burst_detector(drug_features.get('spike_data', {}),
                                      min_burstlet_participation=0.05, **bp)
        drug_pd = drug_nb.get('plot_data', {})
        drug_events = {lvl: drug_nb.get(lvl, {}).get('events', [])
                       for lvl in ['burstlets', 'network_bursts', 'superbursts']}

        sim_ratios = drug_info.get('sim_ratios', {}) or {}
        exp_ratios = drug_info.get('exp_ratios', {}) or {}

        fig = plt.figure(figsize=(16, 17))
        gs = fig.add_gridspec(5, 1, height_ratios=[3, 2, 3, 2, 2.2])
        ax_b_raster = fig.add_subplot(gs[0])
        ax_b_net = fig.add_subplot(gs[1], sharex=ax_b_raster)
        ax_d_raster = fig.add_subplot(gs[2], sharex=ax_b_raster)
        ax_d_net = fig.add_subplot(gs[3], sharex=ax_b_raster)
        ax_bar = fig.add_subplot(gs[4])

        fit_str = f"{drug_fit:.1f}" if isinstance(drug_fit, (int, float)) else str(drug_fit)

        # Baseline
        _plot_raster(ax_b_raster, baseline_features.get('spike_data', {}),
                     title=f"[{drug}] fit={fit_str} | Baseline Raster",
                     cell_types=sim_cell_types)
        _overlay_bursts(ax_b_raster, base_events)
        if base_pd:
            _plot_network_signal(ax_b_net, base_pd, title="Baseline Network Activity")
            _overlay_bursts(ax_b_net, base_events)
        else:
            ax_b_net.text(0.5, 0.5, "No baseline burst data",
                          transform=ax_b_net.transAxes, ha='center')

        # Drug
        _plot_raster(ax_d_raster, drug_features.get('spike_data', {}),
                     title=f"[{drug}] Drug Raster", cell_types=sim_cell_types)
        _overlay_bursts(ax_d_raster, drug_events)
        if drug_pd:
            _plot_network_signal(ax_d_net, drug_pd, title=f"[{drug}] Drug Network Activity")
            _overlay_bursts(ax_d_net, drug_events)
        else:
            ax_d_net.text(0.5, 0.5, "No drug burst data",
                          transform=ax_d_net.transAxes, ha='center')
        ax_d_net.set_xlabel("Time (s)")

        # --- Ratio comparison bar chart (sim vs exp, post/pre) ---
        feats = [f for f in _DRUG_RATIO_FEATURE_MAP if f in sim_ratios or f in exp_ratios]
        x = np.arange(len(feats))
        w = 0.38
        sim_vals = [float(sim_ratios.get(f, np.nan)) for f in feats]
        exp_vals = [float(exp_ratios.get(f, np.nan)) for f in feats]
        ax_bar.bar(x - w / 2, sim_vals, w, label='Simulated', color='tab:blue', alpha=0.85)
        ax_bar.bar(x + w / 2, exp_vals, w, label='Experimental', color='tab:orange', alpha=0.85)
        ax_bar.axhline(1.0, color='gray', ls='--', lw=1, alpha=0.7)  # ratio = no change
        for xi, sv, ev in zip(x, sim_vals, exp_vals):
            if np.isfinite(sv):
                ax_bar.text(xi - w / 2, sv, f"{sv:.2f}", ha='center',
                            va='bottom', fontsize=7)
            if np.isfinite(ev):
                ax_bar.text(xi + w / 2, ev, f"{ev:.2f}", ha='center',
                            va='bottom', fontsize=7)
            if np.isfinite(sv) and np.isfinite(ev):
                ax_bar.text(xi, -0.06, f"Δ={abs(sv - ev):.2f}",
                            ha='center', va='top', fontsize=7, color='dimgray',
                            transform=ax_bar.get_xaxis_transform())
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([f.replace('_ratio', '') for f in feats],
                               rotation=15, fontsize=8)
        ax_bar.set_ylabel("post / pre ratio")
        ax_bar.set_title(f"[{drug}] Simulated vs Experimental drug-response ratios "
                         f"(fit={fit_str})", fontsize=10)
        ax_bar.legend(fontsize=8, loc='upper right', framealpha=0.7)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)

        plt.tight_layout()
        try:
            fig.savefig(save_path, dpi=200)
            logger.info(f"Saved drug response plot: {save_path}")
            for ax in (ax_b_raster, ax_b_net, ax_d_raster, ax_d_net):
                ax.set_xlim(0, 30)
            fig.savefig(save_path.replace('.png', '_30s.png'), dpi=200)
        finally:
            plt.close(fig)


# =============================================================================
# DRUG-RESPONSE SCORING (multi-drug optimization)
# =============================================================================

def _build_features_from_drug_pkl(drug_entry: Dict, recording_duration: float,
                                  min_time: Optional[float] = None) -> Dict:
    """Build the {spike_data, n_units, total_spikes, ...} dict the burst
    detector expects from a ``drug_simData[drug]`` entry (spkt/spkid pair).

    Reuses ``extract_simulated_features`` so the time-shift / per-unit dict
    construction stays identical between baseline and drug paths. ``min_time``
    (the network_cool_down, seconds) drops spikes before that time and shifts
    the rest to t=0 — mirroring the baseline path so the drug post/pre ratios
    are computed over the same settled window (the first ``min_time`` s are a
    cool-down and must be excluded).
    """
    return extract_simulated_features(
        {'spkt': drug_entry.get('spkt', []), 'spkid': drug_entry.get('spkid', [])},
        recording_duration=recording_duration,
        min_time=min_time if (min_time and min_time > 0) else None,
    )


# Burst-detector params for the drug post/pre rate ratios. MUST match the params
# the EXPERIMENTAL ratios were extracted with so simulated and experimental ratios
# are computed on identical footing — see extract_drug_effects.MAIN_BURST_KW /
# PRE_BURST_KW. The MAIN detector (static=40) produces burstlets / network_bursts /
# superbursts; a SECOND detector at the lower static=15 threshold produces the
# pre-burstlets (its 'burstlets' level). Without an explicit base_threshold_static
# the detector defaults to multiplier mode (baseline_val x3), recomputed per
# spike-train, so each condition would get a different threshold and the rate
# ratios would reflect threshold adaptation rather than real change.
_DRUG_BURST_KW = {'base_threshold_static': 40, 'min_burstlet_participation': 0.05}
_DRUG_PRE_BURST_KW = {'base_threshold_static': 15, 'min_burstlet_participation': 0.05}


def _level_stat(nb_result: Dict, level: str, key: str) -> float:
    """Mean of a per-level burst statistic, 0.0 when the level has no events.

    ``level_metrics()`` in the detector returns ``{}`` for a level with zero
    events, so every hop has to be ``.get``-chained.
    """
    return float(
        nb_result.get(level, {}).get('metrics', {}).get(key, {}).get('mean', 0.0)
    )


# Quantile above which a unit is called inhibitory. Matches
# extract_drug_effects.INHIB_QUANTILE so the simulated E/I split is defined the
# same way as the experimental one.
_INHIB_QUANTILE = 0.80


def _classify_top20(spike_data: Dict, recording_duration: float,
                    quantile: float = _INHIB_QUANTILE) -> Dict:
    """Label the top (1-quantile) fraction of units by firing rate as inhibitory.

    Mirrors ``extract_drug_effects.classify_channels_top20``. The simulation does
    know each cell's true type, but the experimental target does not -- its E/I
    split is this firing-rate proxy -- so using the proxy on both sides is what
    makes the exc/inh ratios comparable at all.
    """
    if not spike_data or recording_duration <= 0:
        return {}
    rates = {uid: len(st) / recording_duration for uid, st in spike_data.items()}
    threshold = float(np.quantile(list(rates.values()), quantile))
    return {
        uid: ('inhibitory' if r >= threshold else 'excitatory')
        for uid, r in rates.items()
    }


def _class_mean_rate(spike_data: Dict, classification: Dict, cell_type: str,
                     recording_duration: float) -> float:
    """Mean firing rate (Hz) across units of one class."""
    if recording_duration <= 0 or not classification:
        return 0.0
    members = [uid for uid, c in classification.items()
               if c == cell_type and uid in spike_data]
    if not members:
        return 0.0
    return float(np.mean([len(spike_data[uid]) / recording_duration for uid in members]))


# Every key produced by _ratios_from_features. The degenerate branch and the
# normal branch must return exactly this key set -- _DRUG_RATIO_FEATURE_MAP looks
# them up unconditionally, so a key missing from either branch is a KeyError on
# whichever candidate happens to fall down that path.
_DRUG_RATIO_FEATURE_KEYS = (
    'pop_FR', 'exc_firing_rate', 'inh_firing_rate',
    'network_burst_rate', 'network_burst_duration', 'network_burst_amp',
    'burstlet_rate', 'burstlet_duration', 'burstlet_amp',
    'superburst_rate', 'superburst_duration',
    'pre_burstlet_rate', 'pre_burstlet_duration',
)


def _ratios_from_features(features: Dict, recording_duration: float, burst_params: Dict,
                          classification: Optional[Dict] = None) -> Dict[str, float]:
    """Return the population-level features used for post/pre ratios.

    Runs the burst detector with the same fixed params on every condition AND
    the same the experimental ratios were extracted with, so pre/post and sim/exp
    are all commensurable: ``_DRUG_BURST_KW`` (static=40) for burstlets /
    network_bursts / superbursts, and ``_DRUG_PRE_BURST_KW`` (static=15) for
    pre-burstlets. Caller-supplied ``burst_params`` override these defaults.

    ``classification`` must be derived from the BASELINE condition and passed in
    unchanged for the drug condition, so a unit's E/I label is fixed by its
    pre-drug behaviour -- the same convention extract_drug_effects uses. Deriving
    it separately per condition would let disinhibition reshuffle the labels and
    make the exc/inh ratios meaningless.

    Burst amplitude is divided by unit count because the detector's PFR is a raw
    sum over units, not a per-unit mean. It cancels in a post/pre ratio either
    way, but keeping it normalised here means the value logged in the fitness
    json is directly comparable to the baseline scorer's.
    """
    if features['n_units'] == 0 or features['total_spikes'] == 0:
        return {k: 0.0 for k in _DRUG_RATIO_FEATURE_KEYS}

    n_units = max(int(features['n_units']), 1)
    spike_data = features['spike_data']
    pop_FR = features['total_spikes'] / max(n_units * recording_duration, 1e-9)

    cls = classification or {}
    bp = burst_params or {}
    nb = _run_burst_detector(spike_data, **{**_DRUG_BURST_KW, **bp})
    pre_nb = _run_burst_detector(spike_data, **{**_DRUG_PRE_BURST_KW, **bp})

    def _rate(d, level): return float(d.get(level, {}).get('metrics', {}).get('rate', 0.0))
    def _amp(d, level): return _level_stat(d, level, 'burst_peak') / n_units

    return {
        'pop_FR':                 float(pop_FR),
        'exc_firing_rate':        _class_mean_rate(spike_data, cls, 'excitatory', recording_duration),
        'inh_firing_rate':        _class_mean_rate(spike_data, cls, 'inhibitory', recording_duration),

        'network_burst_rate':     _rate(nb, 'network_bursts'),
        'network_burst_duration': _level_stat(nb, 'network_bursts', 'duration'),
        'network_burst_amp':      _amp(nb, 'network_bursts'),

        'burstlet_rate':          _rate(nb, 'burstlets'),
        'burstlet_duration':      _level_stat(nb, 'burstlets', 'duration'),
        'burstlet_amp':           _amp(nb, 'burstlets'),

        'superburst_rate':        _rate(nb, 'superbursts'),
        'superburst_duration':    _level_stat(nb, 'superbursts', 'duration'),

        'pre_burstlet_rate':      _rate(pre_nb, 'burstlets'),
        'pre_burstlet_duration':  _level_stat(pre_nb, 'burstlets', 'duration'),
    }


# Below this, a baseline value is treated as "this network had none of these
# events" rather than as a divisor.
_RATIO_DENOM_FLOOR = 1e-6


def _safe_sim_ratio(num: float, denom: float) -> Optional[float]:
    """post/pre ratio, or None when the baseline value is too small to divide by.

    The old code did ``num / max(denom, 1e-9)``, which has two failure modes:

      * numerator > 0 -> a ratio around 1e9, indistinguishable once the loss
        saturates from a genuine enormous drug effect. Real ratios up to 4.7e9
        appear in the smoke_v4 batch.
      * numerator == 0 -> a ratio of exactly 0.0, which then compares as a
        PERFECT MATCH against any experimental ratio that is also 0.0. That is a
        0/0 scored as success: every candidate with no baseline superbursts
        collected the superburst weight for free, regardless of what the drug did.

    Returning None lets the caller drop the feature and renormalise the remaining
    weights, so an undefined ratio counts as "no information" rather than either
    "perfectly wrong" or "perfectly right".

    NOTE this changes drug scores on the experimental path too: a trial that
    previously scored 24899.958 now scores 27666.62, purely because the free
    superburst credit (weight 0.10) is no longer granted. Baseline-only runs are
    unaffected -- compute_drug_response_score returns early when drugs is empty.
    """
    if denom is None or not np.isfinite(denom) or abs(denom) < _RATIO_DENOM_FLOOR:
        return None
    ratio = num / denom
    return float(ratio) if np.isfinite(ratio) else None


_DRUG_ERR_EPS = 1e-3

# Scale for a *scored* drug loss. The per-feature loss is already normalized to
# [0, 1] (0 == perfect match), so a fully-scored drug term must land on the same
# 0..100 scale as every baseline component (unit_metrics, burst scorers all
# multiply their normalized error by 100 -- e.g. fitnessFunc_v2.py:395, :832,
# :1252). Using ``max_score`` (30000) here instead was a bug: a drug sim that ran
# fine but matched maximally badly scored 30000, indistinguishable in magnitude
# from a dead-network dealbreaker, so drug_response silently became ~97% of
# fitness despite a nominal weight of 0.40. GENUINE failures (missing pkl,
# crashed sim, no schema, dead baseline) still return ``max_score`` = 30000, the
# dealbreaker sentinel, matching the baseline dead-network convention at :1115.
_DRUG_LOSS_SCALE = 100.0


def _drug_feature_error(sim_r: float, exp_r: float, max_dev: float, loss: str) -> float:
    """Normalized per-feature drug loss in [0, 1]. 0 == exact match.

    Two forms, selected by ``fit_schema['drug_response']['loss']``:

    ``'clipped'`` (default, legacy)
        ``min(|sim - exp| / max_dev, 1.0)``. Kept as the default so existing
        experimental-target runs score identically.

    ``'log_soft'``
        Distance measured in log space -- ratios are multiplicative, so sim=0.5
        against exp=2.0 is as wrong as sim=2.0 against exp=8.0 -- then softly
        saturated with ``d/(1+d)``, which is strictly increasing on [0, inf).

    The clipped form has a dead zone that is the measured cause of the flat drug
    gradient: with max_dev = |exp - 1|, EVERY simulated ratio from 0 up to 1
    yields exactly 1.0, so "the drug did nothing" and "the drug abolished
    bursting entirely" are indistinguishable. In smoke_v4 that pinned the
    bicuculline term at exactly 24899.958 for 81 of 99 trials, because 91 of 99
    produced zero detected bursts under GABA block. A term with no gradient
    cannot steer a search, which is the whole point of the drug run.
    """
    if loss != 'log_soft':
        return min(abs(float(sim_r) - float(exp_r)) / max(float(max_dev), 1e-9), 1.0)

    num = max(float(sim_r), 0.0) + _DRUG_ERR_EPS
    den = max(float(exp_r), 0.0) + _DRUG_ERR_EPS
    scale = max(float(np.log1p(max(float(max_dev), 0.0))), 1e-9)
    d = abs(float(np.log(num / den))) / scale
    if not np.isfinite(d):
        return 1.0
    return float(d / (1.0 + d))


# Schema-feature name → (sim/exp feature key, exp h5 dataset name).
# fit_schema names end in '_ratio'; we strip that to look up the underlying feature.
# Every value on the left must exist in _DRUG_RATIO_FEATURE_KEYS, and every name
# on the right must exist as a dataset in /drug_effects/<drug>/ of the target h5 --
# a schema metric whose h5 dataset is missing is silently skipped at scoring time.
_DRUG_RATIO_FEATURE_MAP = {
    'pop_FR_ratio':                 ('pop_FR',                 'pop_FR_ratio'),
    'exc_firing_rate_ratio':        ('exc_firing_rate',        'exc_firing_rate_ratio'),
    'inh_firing_rate_ratio':        ('inh_firing_rate',        'inh_firing_rate_ratio'),

    'network_burst_rate_ratio':     ('network_burst_rate',     'network_burst_rate_ratio'),
    'network_burst_duration_ratio': ('network_burst_duration', 'network_burst_duration_ratio'),
    'network_burst_amp_ratio':      ('network_burst_amp',      'network_burst_amp_ratio'),

    'burstlet_rate_ratio':          ('burstlet_rate',          'burstlet_rate_ratio'),
    'burstlet_duration_ratio':      ('burstlet_duration',      'burstlet_duration_ratio'),
    'burstlet_amp_ratio':           ('burstlet_amp',           'burstlet_amp_ratio'),

    'superburst_rate_ratio':        ('superburst_rate',        'superburst_rate_ratio'),
    'superburst_duration_ratio':    ('superburst_duration',    'superburst_duration_ratio'),

    'pre_burstlet_rate_ratio':      ('pre_burstlet_rate',      'pre_burstlet_rate_ratio'),
    'pre_burstlet_duration_ratio':  ('pre_burstlet_duration',  'pre_burstlet_duration_ratio'),
}


def compute_drug_response_score(
    drugs: List[str],
    drug_simData: Dict[str, Dict],
    baseline_features: Dict,
    reference_data_path: str,
    fit_schema: Optional[Dict[str, Any]],
    recording_duration: float,
    burst_params: Optional[Dict] = None,
    max_score: float = 30000.0,
    network_cool_down: float = 0.0,
) -> Dict[str, Any]:
    """Score simulated post/pre rate ratios vs experimental drug ratios.

    For each drug in ``drugs``:
      1. Build features from its spkt/spkid in ``drug_simData[drug]``.
      2. Compute simulated post/pre ratios for the features named in the
         ``drug_response`` schema component.
      3. Read experimental ratios from
         ``<reference_h5>/drug_effects/<drug>/<feature>_ratio``.
      4. Per-feature loss = ``min(|sim - exp| / max_dev, 1.0) * max_score``,
         weighted by the schema metric weight. Loss is averaged across
         drug-metric pairs that match; each drug's loss is averaged across
         features and a missing-or-failed drug gets ``max_score``.

    Returns
    -------
    dict
        ``{'fit': float, 'per_drug': {drug: {sim_ratios, exp_ratios, fit, reason?}}}``
    """
    if not drugs:
        return {'fit': 0.0, 'per_drug': {}}

    schema_metrics = {}
    drug_loss_form = 'clipped'
    if isinstance(fit_schema, dict):
        _drug_block = fit_schema.get('drug_response', {}) or {}
        schema_metrics = _drug_block.get('metrics', {}) or {}
        # Opt-in per schema so experimental-target runs keep the legacy loss.
        drug_loss_form = str(_drug_block.get('loss', 'clipped'))
    if not schema_metrics:
        # No schema entry for drug_response → cannot score; treat as max_score
        # so the optimizer can't game the missing weight.
        return {'fit': float(max_score), 'per_drug': {d: {'fit': max_score,
                                                          'reason': 'no drug_response schema'}
                                                     for d in drugs}}

    # E/I labels are fixed by the BASELINE condition and reused unchanged for every
    # drug, matching how extract_drug_effects fixes them from the PRE recording.
    baseline_classification = _classify_top20(
        baseline_features.get('spike_data', {}), recording_duration
    )

    # Baseline rate features — same shape on both sides.
    baseline_rates = _ratios_from_features(baseline_features, recording_duration,
                                           burst_params or {},
                                           classification=baseline_classification)

    per_drug = {}
    drug_losses: List[float] = []
    for drug in drugs:
        entry = (drug_simData or {}).get(drug)
        if not entry or entry.get('_failure'):
            per_drug[drug] = {'fit': float(max_score),
                              'reason': entry.get('_failure_reason') if entry else 'missing'}
            drug_losses.append(max_score)
            continue

        drug_features = _build_features_from_drug_pkl(entry, recording_duration,
                                                      min_time=network_cool_down)
        drug_rates = _ratios_from_features(drug_features, recording_duration,
                                           burst_params or {},
                                           classification=baseline_classification)

        # Simulated post/pre ratios. Features whose baseline value is ~0 are
        # dropped rather than divided by, and recorded so the omission is visible
        # in trial_*_fitness.json instead of silently reweighting the drug term.
        sim_ratios: Dict[str, float] = {}
        skipped_features: Dict[str, str] = {}
        for feat, (rate_key, _) in _DRUG_RATIO_FEATURE_MAP.items():
            if feat not in schema_metrics:
                continue
            ratio = _safe_sim_ratio(drug_rates.get(rate_key, 0.0),
                                    baseline_rates.get(rate_key, 0.0))
            if ratio is None:
                skipped_features[feat] = (
                    f'baseline {rate_key}={baseline_rates.get(rate_key, 0.0):.3g} '
                    f'below {_RATIO_DENOM_FLOOR:g}'
                )
            else:
                sim_ratios[feat] = ratio

        # Experimental ratios from the reference h5
        exp_ratios: Dict[str, float] = {}
        try:
            with h5py.File(reference_data_path, 'r') as _f:
                grp = _f.get(f'drug_effects/{drug}')
                if grp is None:
                    per_drug[drug] = {'fit': float(max_score),
                                      'reason': f'no /drug_effects/{drug} in {reference_data_path}'}
                    drug_losses.append(max_score)
                    continue
                for feat, (_, h5_name) in _DRUG_RATIO_FEATURE_MAP.items():
                    if feat in schema_metrics and h5_name in grp:
                        exp_ratios[feat] = float(grp[h5_name][()])
        except Exception as _e:
            per_drug[drug] = {'fit': float(max_score),
                              'reason': f'h5 read error: {type(_e).__name__}: {_e}'}
            drug_losses.append(max_score)
            continue

        # A schema metric with no matching dataset is skipped silently below,
        # which shrinks the effective drug weight instead of failing. Say so.
        _missing = [f for f in schema_metrics
                    if f in _DRUG_RATIO_FEATURE_MAP and f not in exp_ratios]
        if _missing:
            logger.warning(
                f"[{drug}] schema metrics with no dataset in "
                f"{reference_data_path}:/drug_effects/{drug} (they will not be "
                f"scored): {_missing}"
            )

        # Weighted loss over every feature the target actually defines. A feature
        # whose ratio is undefined (degenerate baseline) is charged FULL loss, not
        # dropped.
        #
        # Dropping it and renormalising over the survivors is a loophole, and it
        # wins: a candidate whose baseline produces no bursts makes all five burst
        # features unscorable and is then graded only on the three firing-rate
        # features -- 35% of the objective. Measured in gen_7/gen_12 of the first
        # theory run, that shortcut took the two BEST drug scores (9415, 12991),
        # beating gen_9 (15580) which actually reproduced post-drug bursting and
        # was graded on all eight. Charging full loss keeps the denominator fixed
        # so "I made this unmeasurable" can never beat "I got it wrong".
        feat_w_total = 0.0
        feat_loss_acc = 0.0
        unscorable_weight = 0.0
        for feat in exp_ratios:
            mcfg = schema_metrics.get(feat, {}) or {}
            w = float(mcfg.get('weight', 0.0))
            if w <= 0:
                continue
            mx = float(mcfg.get('max_dev', 1.0))
            if feat in sim_ratios:
                err = _drug_feature_error(sim_ratios[feat], exp_ratios[feat], mx, drug_loss_form)
            else:
                err = 1.0
                unscorable_weight += w
            feat_loss_acc += w * err
            feat_w_total += w

        if feat_w_total <= 0:
            # Nothing scorable → degenerate; max score. Two very different causes,
            # and reporting the wrong one sends the reader hunting a config bug
            # when the real answer is that the candidate network is dead.
            if skipped_features and not sim_ratios:
                _reason = (f'no scorable features: baseline produced no activity '
                           f'(total_spikes={baseline_features.get("total_spikes", 0)}) '
                           f'so every ratio is undefined')
            elif not exp_ratios:
                _reason = (f'no scorable features: none of the schema metrics exist as '
                           f'datasets in /drug_effects/{drug}')
            else:
                _reason = 'no scorable features: all matched metrics have zero weight'
            drug_fit = float(max_score)
            per_drug[drug] = {'fit': drug_fit, 'reason': _reason,
                              'sim_ratios': sim_ratios, 'exp_ratios': exp_ratios,
                              'skipped_features': skipped_features}
        else:
            # Normalized [0,1] loss -> 0..100, same scale as baseline components.
            # NOT max_score (30000); that is the dealbreaker sentinel, reserved
            # for the failure branches above. See _DRUG_LOSS_SCALE.
            drug_fit = float(_DRUG_LOSS_SCALE * (feat_loss_acc / feat_w_total))
            per_drug[drug] = {'fit': drug_fit,
                              'sim_ratios': sim_ratios, 'exp_ratios': exp_ratios,
                              'skipped_features': skipped_features,
                              # Fraction of the drug weight charged full loss because
                              # its ratio was undefined. >0 means the candidate is
                              # being penalised for an unmeasurable baseline, not for
                              # a wrong drug response.
                              'unscorable_weight_fraction': float(unscorable_weight),
                              'scored_weight_fraction': float(feat_w_total)}

        # Raw per-condition values, not just their ratio. Without these a ratio of
        # 0.0 is ambiguous: it can mean the drug network saturated into tonic
        # firing (spikes way up, detector finds no discrete bursts) or that it
        # went silent (spikes ~0). Those need opposite fixes, and ratios alone
        # cannot tell them apart -- which is what made smoke_v4 hard to diagnose.
        per_drug[drug]['loss_form'] = drug_loss_form
        per_drug[drug]['baseline_condition'] = {
            'n_units': int(baseline_features.get('n_units', 0)),
            'total_spikes': int(baseline_features.get('total_spikes', 0)),
        }
        per_drug[drug]['drug_condition'] = {
            'n_units': int(drug_features.get('n_units', 0)),
            'total_spikes': int(drug_features.get('total_spikes', 0)),
        }
        per_drug[drug]['baseline_rates'] = {k: float(v) for k, v in baseline_rates.items()}
        per_drug[drug]['drug_rates'] = {k: float(v) for k, v in drug_rates.items()}
        drug_losses.append(drug_fit)

    overall = float(np.mean(drug_losses)) if drug_losses else 0.0
    overall = min(overall, max_score) if np.isfinite(overall) else max_score
    return {'fit': overall, 'per_drug': per_drug}


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

    # --- weights: read top-level weights directly from fit_schema ---
    if weights is None:
        _active_schema = kwargs.get('fit_schema', None) or DEFAULT_FIT_SCHEMA
        weights = {
            name: float(comp.get('weight', 0.0))
            for name, comp in _active_schema.items()
            if isinstance(comp, dict)
        }
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()} if total_w > 0 else weights

    if config is None:
        print()
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
                        kwargs['simCfg'] = sim.cfg.todict().copy()
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
        fit_schema = kwargs.get('fit_schema', None)

        logger.info("Computing unit metrics")
        unit_results = compute_unit_score(
            sim_features,
            exp_data,
            recording_duration,
            sim_excit_units=kwargs.get('excit_units'),
            sim_inhib_units=kwargs.get('inhib_units'),
            fit_schema=fit_schema,
            pop_data=kwargs.get('popData'),
        )

        logger.info("Computing synchrony metrics")
        sync_results = compute_synchrony_score(
            sim_features['spike_data'], exp_data,
            bin_size=config.get('synchrony', {}).get('bin_size', 0.01),
            recording_duration=recording_duration,
            fit_schema=fit_schema,
        )

        logger.info("Computing hierarchical burst metrics")
        burst_params = config.get('hierarchical_bursts', {})
        burst_params = {}
        use_simple_burst_scoring = kwargs.get('use_simple_burst_scoring', False)
        use_v2_burst_scoring = kwargs.get('use_v2_burst_scoring', True)
        # import pdb; pdb.set_trace()
        if use_v2_burst_scoring:
            logger.info("Using simple burst scoring v2 (schema-driven weights for count/timing/spike-width)")
            burst_results = compute_hierarchical_burst_scores_simple_v2(
                sim_features['spike_data'], exp_data['spike_data'],
                recording_duration, burst_params,
                fit_schema=fit_schema,
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
                fit_schema=fit_schema,
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

        # ----------------------------------------------------------------
        # 4b. Drug-response scoring (only when --drugs was used)
        # ----------------------------------------------------------------
        drugs_requested = kwargs.get('drugs', []) or []
        drug_simData = kwargs.get('drug_simData', {}) or {}
        if drugs_requested:
            drug_results = compute_drug_response_score(
                drugs=drugs_requested,
                drug_simData=drug_simData,
                baseline_features=sim_features,
                reference_data_path=reference_data_path,
                fit_schema=fit_schema,
                recording_duration=recording_duration,
                burst_params=burst_params,
                max_score=max_score,
                network_cool_down=network_cool_down,
            )
            drug_response_score = _clamp(drug_results['fit'])
        else:
            drug_results = {'fit': 0.0, 'per_drug': {}}
            drug_response_score = 0.0

        fitness = (
            weights.get('unit_metrics',    0.0) * unit_score +
            weights.get('synchrony',       0.0) * sync_score +
            weights.get('pre_burstlets',   0.0) * pre_burstlet_score +
            weights.get('burstlets',       0.0) * burstlet_score +
            weights.get('network_bursts',  0.0) * network_burst_score +
            weights.get('superbursts',     0.0) * superburst_score +
            weights.get('drug_response',   0.0) * drug_response_score
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
                'drug_response': {'fit': float(drug_response_score),
                                  'per_drug': drug_results.get('per_drug', {})},
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
                    sim_excit_units=kwargs.get('excit_units'),
                    sim_inhib_units=kwargs.get('inhib_units'),
                    pop_data=kwargs.get('popData'),
                )
            except Exception as e:
                logger.warning(f"Plotting failed: {e}")

            # Drug-response plots: one figure per drug (baseline vs drug rasters,
            # network activity, and the simulated-vs-experimental ratio bars).
            if drugs_requested:
                try:
                    plot_drug_response(
                        drugs=drugs_requested,
                        drug_simData=drug_simData,
                        baseline_features=sim_features,
                        drug_results=drug_results,
                        recording_duration=recording_duration,
                        save_path_prefix=candidate_path,
                        burst_params=burst_params,
                        baseline_burst_results=burst_results,
                        pop_data=kwargs.get('popData'),
                        network_cool_down=network_cool_down,
                    )
                except Exception as e:
                    logger.warning(f"Drug-response plotting failed: {e}")

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
