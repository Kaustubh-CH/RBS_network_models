"""Five population metrics, defined ONCE and computed identically for both
simulated and experimental pre/post data.

This module is the single place each of the five features is defined. The same
``compute_five_features`` is applied to all four datasets (experimental pre,
experimental post, simulated pre, simulated post) so the change vectors line up
by construction.

Inputs everywhere
-----------------
``units``  : {unit_id -> np.ndarray of spike times in SECONDS}. Include silent
             units (empty arrays) so the population size is correct — pop_FR and
             burst participation both divide by ``len(units)``.
``T``      : window length in seconds. Spikes should already be truncated to
             [0, T] (use ``extract_drug_effects._truncate``).

Features
--------
pop_fr_hz            mean population firing rate = total_spikes / (n_units * T)
burst_freq_per_min   network bursts per minute = n_network_bursts / T * 60
synchrony_index      CV of the binned population spike-count signal,
                     std(c)/mean(c) over fixed ``SYNC_BIN_MS`` bins on [0, T]
                     (burst-detector independent; 0 = uniform, higher = burstier)
ibi_s                mean inter-burst interval = mean(diff(network-burst starts))
burst_duration_s     mean network-burst duration

Network bursts come from the SAME detector + kwargs used for the experimental
drug targets (``parameter_free_burst_detector.compute_network_bursts`` with the
production main pass), so "network burst" means the same thing on both sides.
"""
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent  # .../networkSimulations
sys.path.insert(0, str(_REPO_ROOT / "MEA_Analysis" / "IPNAnalysis"))
from parameter_free_burst_detector import compute_network_bursts  # noqa: E402

logger = logging.getLogger("five_feature_metrics")

EPS = 1e-9
# Network-burst detector kwargs: the production main pass. IDENTICAL for sim and
# exp (matches extract_drug_effects.MAIN_BURST_KW), so "network burst" is one
# definition everywhere.
BURST_KW = dict(base_threshold_static=40, min_burstlet_participation=0.05)
# Fixed bin for the population-rate CV synchrony index. Defined once here.
SYNC_BIN_MS = 20.0

# Canonical order + display labels for the five features.
FEATURE_KEYS = (
    "pop_fr_hz",
    "burst_freq_per_min",
    "synchrony_index",
    "ibi_s",
    "burst_duration_s",
)
FEATURE_LABELS = {
    "pop_fr_hz":          "mean pop FR (Hz)",
    "burst_freq_per_min": "burst freq (/min)",
    "synchrony_index":    "synchrony index (pop-rate CV)",
    "ibi_s":              "inter-burst interval (s)",
    "burst_duration_s":   "burst duration (s)",
}


def _pop_fr_hz(units: Dict[str, np.ndarray], T: float) -> float:
    if not units or T <= 0:
        return 0.0
    total = sum(len(t) for t in units.values())
    return total / (len(units) * T)


def _synchrony_index(units: Dict[str, np.ndarray], T: float) -> float:
    """CV (std/mean) of the binned population spike-count signal over [0, T]."""
    if not units or T <= 0:
        return 0.0
    arrs = [np.asarray(t, dtype=float) for t in units.values() if len(t)]
    if not arrs:
        return 0.0
    all_spikes = np.concatenate(arrs)
    bin_s = SYNC_BIN_MS / 1000.0
    edges = np.arange(0.0, T + bin_s, bin_s)
    if edges.size < 2:
        return 0.0
    counts, _ = np.histogram(all_spikes, bins=edges)
    m = float(counts.mean())
    if m <= 0:
        return 0.0
    return float(counts.std() / m)


def compute_five_features(units: Dict[str, np.ndarray], T: float, label: str = "") -> dict:
    """Compute the five population features for one dataset.

    Returns a dict with the five FEATURE_KEYS plus bookkeeping fields prefixed
    with '_' (n_network_bursts, n_units, total_spikes, T_seconds).
    """
    units = {u: np.asarray(t, dtype=float) for u, t in units.items()}
    pop_fr = _pop_fr_hz(units, T)
    sync = _synchrony_index(units, T)

    n_bursts = 0
    dur_mean = 0.0
    ibi_mean = 0.0
    if units and any(len(t) for t in units.values()):
        nb = compute_network_bursts(SpikeTimes=units, plot=False, verbose=False, **BURST_KW)
        if "error" in nb:
            logger.warning("[%s] burst detector returned %s -> burst features = 0",
                           label, nb["error"])
        else:
            level = nb.get("network_bursts", {})
            events = level.get("events", [])
            metrics = level.get("metrics", {})
            n_bursts = len(events)
            dur_mean = float(metrics.get("duration", {}).get("mean", 0.0))
            ibi_mean = float(metrics.get("inter_event_interval", {}).get("mean", 0.0))

    burst_freq = (n_bursts / T) * 60.0 if T > 0 else 0.0

    feats = {
        "pop_fr_hz":          float(pop_fr),
        "burst_freq_per_min": float(burst_freq),
        "synchrony_index":    float(sync),
        "ibi_s":              float(ibi_mean),
        "burst_duration_s":   float(dur_mean),
        "_n_network_bursts":  int(n_bursts),
        "_n_units":           len(units),
        "_total_spikes":      int(sum(len(t) for t in units.values())),
        "_T_seconds":         float(T),
    }
    logger.info(
        "[%s] pop_FR=%.4f Hz  burst_freq=%.3f/min  sync=%.4f  IBI=%.3f s  "
        "dur=%.3f s  (%d net-bursts, %d units, %d spikes, T=%.2fs)",
        label, pop_fr, burst_freq, sync, ibi_mean, dur_mean,
        n_bursts, len(units), feats["_total_spikes"], T,
    )
    return feats


def ratios_from_features(pre: dict, post: dict) -> Dict[str, float]:
    """post/pre ratio for each of the five features (eps-floored denominator)."""
    return {
        k: float(post[k]) / max(float(pre[k]), EPS)
        for k in FEATURE_KEYS
    }
