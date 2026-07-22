"""
Fitness Function v2 — Drug-Response Ratio
=========================================

Scores how a *simulated* network reacts to a drug, by comparing the simulated
post/pre ratios of a set of population features to the **experimental**
drug-response ratios stamped into a baseline target h5 by
``_scripts/extract_drug_effects.py`` (the same ratios
``_scripts/plot_drug_ratio_histograms.py`` visualises).

Two simulated networks are required:

    baseline (previous) sim   — fixed; "already set, no change"
    post-drug (current) sim   — the candidate being optimised

For each, the same population features are extracted (E/I-split firing rates +
hierarchical burst rates/durations + mean participation). The simulated
drug-response ratio is ``post / pre`` per feature, and the fitness is the
schema-weighted squared log2 distance between the simulated ratio and the
experimental ratio::

    error_feature = (log2(sim_ratio) - log2(exp_ratio)) ** 2
    fitness       = score_scale * Σ_f w_f · error_f   (lower is better)

Both the feature extraction (burst-detector kwargs, E/I split quantile) and the
ratio feature list mirror ``extract_drug_effects.py`` so the simulated and
experimental sides are computed identically.

Author: Auto-generated
Date: May 2026
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

# --- Make burst detector + sibling helpers importable -----------------------
_script_dir = Path(__file__).resolve().parent
_mea_ipn_dir = Path("/pscratch/sd/k/ktub1999/networkSimulations/MEA_Analysis/IPNAnalysis")
for _d in [_script_dir, _mea_ipn_dir]:
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

# Reuse the simulated-data extraction + burst-detector wrapper from v2 so this
# function and the plain v2 function read NetPyNE output identically.
try:
    from .fitnessFunc_v2 import (
        extract_simulated_features,
        get_sim_data_from_call_stack,
        get_sim_data_from_pkl,
        _run_burst_detector,
        load_experimental_h5,
        _plot_raster,
        _plot_network_signal,
        _overlay_bursts,
    )
except ImportError:
    from fitnessFunc_v2 import (  # type: ignore
        extract_simulated_features,
        get_sim_data_from_call_stack,
        get_sim_data_from_pkl,
        _run_burst_detector,
        load_experimental_h5,
        _plot_raster,
        _plot_network_signal,
        _overlay_bursts,
    )

try:
    from netpyne import sim
except ImportError:
    sim = None

# --- Schema --------------------------------------------------------------------
try:
    from fitness_schema.schema_v2_drug import (
        DEFAULT_CONFIG,
        fit_schema as DEFAULT_FIT_SCHEMA,
        RATIO_FEATURES,
        MAIN_BURST_KW,
        PRE_BURST_KW,
        INHIB_QUANTILE,
        EPS,
        PARAM_PENALTY_EXCLUDE,
        PARAM_PENALTY_DENYLIST,
        DRUG_PARAM_EXCLUDE,
        exclude_params_for_drug,
    )
except ImportError:
    try:
        _schema_dir = _script_dir.parent / "fitness_schema"
        if str(_schema_dir) not in sys.path:
            sys.path.insert(0, str(_schema_dir))
        from schema_v2_drug import (  # type: ignore
            DEFAULT_CONFIG,
            fit_schema as DEFAULT_FIT_SCHEMA,
            RATIO_FEATURES,
            MAIN_BURST_KW,
            PRE_BURST_KW,
            INHIB_QUANTILE,
            EPS,
            PARAM_PENALTY_EXCLUDE,
            PARAM_PENALTY_DENYLIST,
            DRUG_PARAM_EXCLUDE,
            exclude_params_for_drug,
        )
    except ImportError:
        # Minimal fallback defaults (kept in sync with schema_v2_drug.py)
        EPS = 1e-9
        MAIN_BURST_KW = dict(base_threshold_static=40, min_burstlet_participation=0.05)
        PRE_BURST_KW = dict(base_threshold_static=15, min_burstlet_participation=0.05)
        INHIB_QUANTILE = 0.80
        RATIO_FEATURES = (
            "pop_FR_ratio", "exc_firing_rate_ratio", "inh_firing_rate_ratio",
            "burstlet_rate_ratio", "burstlet_duration_ratio",
            "network_burst_rate_ratio", "network_burst_duration_ratio",
            "superburst_rate_ratio", "superburst_duration_ratio",
            "pre_burstlet_rate_ratio", "pre_burstlet_duration_ratio",
            "mean_participation_ratio",
        )
        PARAM_PENALTY_EXCLUDE = ('weightIE_GABA', 'weightII_GABA')
        PARAM_PENALTY_DENYLIST = (
            'duration', 'duration_seconds', 'network_cool_down',
            'simLabel', 'seeds', 'filename', 'saveFolder', 'dt', 'recordStep',
        )
        DRUG_PARAM_EXCLUDE = {
            'bicuculline': ('weightIE_GABA', 'weightII_GABA'),
            'ap5_nbqx': ('weightEE_AMPA', 'weightEI_AMPA', 'weightEE_NMDA', 'weightEI_NMDA'),
        }

        def exclude_params_for_drug(drug_name, default=PARAM_PENALTY_EXCLUDE):
            if not drug_name:
                return tuple(default)
            key = ''.join(ch for ch in str(drug_name).lower() if ch.isalnum())
            for k, v in DRUG_PARAM_EXCLUDE.items():
                if ''.join(ch for ch in k.lower() if ch.isalnum()) == key:
                    return tuple(v)
            if any(t in key for t in ('ap5', 'apv', 'nbqx', 'cnqx', 'dnqx', 'kynurenic')):
                return DRUG_PARAM_EXCLUDE['ap5_nbqx']
            if any(t in key for t in ('bicuculline', 'bic', 'gabazine', 'sr95531', 'picrotoxin', 'ptx')):
                return DRUG_PARAM_EXCLUDE['bicuculline']
            return tuple(default)
        DEFAULT_CONFIG = {
            'main_burst_kwargs': dict(MAIN_BURST_KW),
            'pre_burstlet_kwargs': dict(PRE_BURST_KW),
            'inhib_quantile': INHIB_QUANTILE,
            'eps': EPS, 'score_scale': 100.0, 'drug_name': None,
            'recording_duration': 300.0, 'max_fitness': 1_000_000.0,
            'config_penalty_weight': 1.0, 'config_penalty_scale': 1.0,
            'exclude_params': None,
        }
        DEFAULT_FIT_SCHEMA = {'drug_ratios': {'weight': 1.0, 'metrics': {
            f: {'weight': 1.0, 'max_val': 100.0} for f in RATIO_FEATURES}}}

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _resolve_fit_schema(schema_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if isinstance(schema_override, dict):
        return schema_override
    return DEFAULT_FIT_SCHEMA


# =============================================================================
# SIMULATED-NETWORK FEATURE EXTRACTION
# =============================================================================

def _safe_ratio(num: float, denom: float) -> float:
    """post / pre with an epsilon floor on the denominator (matches
    extract_drug_effects._safe_ratio)."""
    return float(num) / max(float(denom), EPS)


def _level_rate(nb_result: dict, level: str) -> float:
    return float(nb_result.get(level, {}).get("metrics", {}).get("rate", 0.0))


def _level_duration_mean(nb_result: dict, level: str) -> float:
    return float(nb_result.get(level, {}).get("metrics", {}).get("duration", {}).get("mean", 0.0))


def _level_participation_mean(nb_result: dict, level: str) -> float:
    return float(nb_result.get(level, {}).get("metrics", {}).get("participation", {}).get("mean", 0.0))


def _pop_firing_rate_hz(units: Dict[int, np.ndarray], T: float) -> float:
    """Population firing rate = total_spikes / (n_units * T) — matches
    extract_drug_effects._pop_firing_rate_hz."""
    if not units or T <= 0:
        return 0.0
    total = sum(len(t) for t in units.values())
    return total / (len(units) * T)


def _class_mean_firing_rate_hz(
    units: Dict[int, np.ndarray], classification: Dict[int, str], cell_type: str, T: float
) -> float:
    if T <= 0:
        return 0.0
    members = [u for u, c in classification.items() if c == cell_type and u in units]
    if not members:
        return 0.0
    return float(np.mean([len(units[u]) / T for u in members]))


def classify_units_top_quantile(
    units: Dict[int, np.ndarray], T: float, quantile: float = INHIB_QUANTILE
) -> Tuple[Dict[int, str], float]:
    """Top ``(1 - quantile)`` fraction of units by firing rate -> inhibitory.

    Mirrors extract_drug_effects.classify_channels_top20 so that the simulated
    E/I-split firing-rate ratios are computed the same way as the experimental
    ones the optimizer is fit against. Returns (classification, threshold_hz).
    """
    if T <= 0 or not units:
        return {}, 0.0
    rates = {u: len(t) / T for u, t in units.items()}
    threshold = float(np.quantile(list(rates.values()), quantile))
    classification = {
        u: ("inhibitory" if r >= threshold else "excitatory") for u, r in rates.items()
    }
    return classification, threshold


def classify_units_from_popdata(pop_data: Dict) -> Dict[int, str]:
    """Ground-truth E/I labels from NetPyNE popData (pop 'I' -> inhibitory)."""
    cls: Dict[int, str] = {}
    for pop_name, pop_info in (pop_data or {}).items():
        ct = 'inhibitory' if pop_name == 'I' else 'excitatory'
        for gid in pop_info.get('cellGids', []):
            cls[int(gid)] = ct
    return cls


def compute_sim_side_features(
    units: Dict[int, np.ndarray],
    T: float,
    classification: Dict[int, str],
    label: str,
    return_details: bool = False,
):
    """Extract the population features we ratio, for one simulated network.

    Returns the same keys as extract_drug_effects._side_features so the
    simulated post/pre ratios are directly comparable to the experimental ones.

    If ``return_details`` is True, returns ``(features, details)`` where
    ``details`` carries the raw main/pre burst-detector results (events +
    plot_data) so the network-comparison figure can be drawn.
    """
    pop_FR = _pop_firing_rate_hz(units, T)
    exc_FR = _class_mean_firing_rate_hz(units, classification, "excitatory", T)
    inh_FR = _class_mean_firing_rate_hz(units, classification, "inhibitory", T)
    logger.info(
        "[%s] pop_FR=%.4f Hz  exc_FR=%.4f Hz  inh_FR=%.4f Hz  (%d units)",
        label, pop_FR, exc_FR, inh_FR, len(units),
    )

    logger.info("[%s] running main burst detector %s ...", label, MAIN_BURST_KW)
    main_nb = _run_burst_detector(units, **MAIN_BURST_KW)
    logger.info("[%s] running pre-burstlet detector %s ...", label, PRE_BURST_KW)
    pre_nb = _run_burst_detector(units, **PRE_BURST_KW)

    features = {
        "pop_FR_hz": pop_FR,
        "exc_firing_rate_hz": exc_FR,
        "inh_firing_rate_hz": inh_FR,
        # main pass
        "burstlet_rate_hz":          _level_rate(main_nb, "burstlets"),
        "burstlet_duration_s":       _level_duration_mean(main_nb, "burstlets"),
        "network_burst_rate_hz":     _level_rate(main_nb, "network_bursts"),
        "network_burst_duration_s":  _level_duration_mean(main_nb, "network_bursts"),
        "superburst_rate_hz":        _level_rate(main_nb, "superbursts"),
        "superburst_duration_s":     _level_duration_mean(main_nb, "superbursts"),
        "mean_participation":        _level_participation_mean(main_nb, "network_bursts"),
        # pre-burstlet pass (permissive threshold)
        "pre_burstlet_rate_hz":      _level_rate(pre_nb, "burstlets"),
        "pre_burstlet_duration_s":   _level_duration_mean(pre_nb, "burstlets"),
    }
    if not return_details:
        return features

    details = {
        'spike_data': units,
        'burst_events': {
            lvl: main_nb.get(lvl, {}).get('events', [])
            for lvl in ('burstlets', 'network_bursts', 'superbursts')
        },
        'plot_data': main_nb.get('plot_data', {}),
    }
    return features, details


def compute_sim_ratios(
    pre_features: Dict[str, float],
    post_features: Dict[str, float],
) -> Dict[str, float]:
    """post/pre ratios, keyed by RATIO_FEATURES (matches
    extract_drug_effects.compute_ratios)."""
    return {
        "pop_FR_ratio":                 _safe_ratio(post_features["pop_FR_hz"], pre_features["pop_FR_hz"]),
        "exc_firing_rate_ratio":        _safe_ratio(post_features["exc_firing_rate_hz"], pre_features["exc_firing_rate_hz"]),
        "inh_firing_rate_ratio":        _safe_ratio(post_features["inh_firing_rate_hz"], pre_features["inh_firing_rate_hz"]),
        "burstlet_rate_ratio":          _safe_ratio(post_features["burstlet_rate_hz"], pre_features["burstlet_rate_hz"]),
        "burstlet_duration_ratio":      _safe_ratio(post_features["burstlet_duration_s"], pre_features["burstlet_duration_s"]),
        "network_burst_rate_ratio":     _safe_ratio(post_features["network_burst_rate_hz"], pre_features["network_burst_rate_hz"]),
        "network_burst_duration_ratio": _safe_ratio(post_features["network_burst_duration_s"], pre_features["network_burst_duration_s"]),
        "superburst_rate_ratio":        _safe_ratio(post_features["superburst_rate_hz"], pre_features["superburst_rate_hz"]),
        "superburst_duration_ratio":    _safe_ratio(post_features["superburst_duration_s"], pre_features["superburst_duration_s"]),
        "pre_burstlet_rate_ratio":      _safe_ratio(post_features["pre_burstlet_rate_hz"], pre_features["pre_burstlet_rate_hz"]),
        "pre_burstlet_duration_ratio":  _safe_ratio(post_features["pre_burstlet_duration_s"], pre_features["pre_burstlet_duration_s"]),
        "mean_participation_ratio":     _safe_ratio(post_features["mean_participation"], pre_features["mean_participation"]),
    }


# =============================================================================
# EXPERIMENTAL DRUG-RATIO LOADING (from target h5 /drug_effects/<drug_name>/)
# =============================================================================

def load_experimental_drug_ratios(
    h5_path: str, drug_name: Optional[str] = None
) -> Tuple[Dict[str, float], str]:
    """Read the experimental post/pre ratios written by extract_drug_effects.py.

    Returns (ratios, resolved_drug_name). If ``drug_name`` is None the first
    group under ``/drug_effects`` is used.
    """
    ratios: Dict[str, float] = {}
    resolved = drug_name or ""
    with h5py.File(h5_path, 'r') as f:
        if 'drug_effects' not in f:
            raise KeyError(
                f"No /drug_effects group in {h5_path}; run extract_drug_effects.py first."
            )
        root = f['drug_effects']
        if drug_name is None:
            keys = list(root.keys())
            if not keys:
                raise KeyError(f"/drug_effects is empty in {h5_path}")
            resolved = keys[0]
            if len(keys) > 1:
                logger.warning("Multiple drug groups %s; using '%s'. Pass drug_name to choose.",
                               keys, resolved)
        if resolved not in root:
            raise KeyError(
                f"/drug_effects/{resolved} not found in {h5_path}. "
                f"Available: {list(root.keys())}"
            )
        grp = root[resolved]
        for feat in RATIO_FEATURES:
            if feat in grp:
                ratios[feat] = float(grp[feat][()])
            else:
                logger.warning("Feature '%s' missing from /drug_effects/%s", feat, resolved)
    return ratios, resolved


# =============================================================================
# SCORING
# =============================================================================

def score_drug_ratios(
    sim_ratios: Dict[str, float],
    exp_ratios: Dict[str, float],
    fit_schema: Optional[Dict[str, Any]] = None,
    score_scale: float = 100.0,
    max_score: float = 30000.0,
) -> Dict[str, Any]:
    """Schema-weighted squared log2-ratio distance between sim and exp ratios.

    For each feature::

        error = (log2(clip(sim_ratio)) - log2(clip(exp_ratio))) ** 2

    log space makes a 2x increase and a 0.5x decrease equidistant from "no
    change" (1.0). The weighted mean error is scaled and clamped to max_score.
    """
    fit_schema = _resolve_fit_schema(fit_schema)
    metrics = fit_schema.get('drug_ratios', {}).get('metrics', {})

    per_feature: Dict[str, Dict[str, float]] = {}
    weighted_sum = 0.0
    weight_total = 0.0

    for feat in RATIO_FEATURES:
        if feat not in sim_ratios or feat not in exp_ratios:
            continue
        sim_r = max(float(sim_ratios[feat]), EPS)
        exp_r = max(float(exp_ratios[feat]), EPS)
        log_err = float((np.log2(sim_r) - np.log2(exp_r)) ** 2)

        cfg = metrics.get(feat, {})
        w = float(cfg.get('weight', 0.0))
        max_val = float(cfg.get('max_val', max_score))
        clipped_err = min(log_err, max_val)

        per_feature[feat] = {
            'sim_ratio': float(sim_ratios[feat]),
            'exp_ratio': float(exp_ratios[feat]),
            'log2_sim': float(np.log2(sim_r)),
            'log2_exp': float(np.log2(exp_r)),
            'log2_error': log_err,
            'weight': w,
        }
        weighted_sum += w * clipped_err
        weight_total += w

    if weight_total > 0:
        total = (weighted_sum / weight_total) * score_scale
    else:
        # Unweighted fallback
        errs = [v['log2_error'] for v in per_feature.values()]
        total = float(np.mean(errs)) * score_scale if errs else max_score

    total = min(float(total), max_score) if np.isfinite(total) else max_score

    return {
        'per_feature': per_feature,
        'n_features_scored': len(per_feature),
        'total_score': total,
    }


# =============================================================================
# CONFIG-DRIFT PENALTY
# =============================================================================

def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def compute_config_penalty(
    baseline_cfg: Dict[str, Any],
    current_cfg: Dict[str, Any],
    param_names: Optional[List[str]] = None,
    exclude: Tuple[str, ...] = PARAM_PENALTY_EXCLUDE,
    penalty_weight: float = 1.0,
    score_scale: float = 100.0,
    eps: float = EPS,
) -> Dict[str, Any]:
    """Penalise drift of the candidate (post-drug) config from the baseline.

    A drug is modelled as changing ONLY the GABA inhibitory weights (``exclude``).
    Every other compared parameter should stay equal to the baseline network, so
    the penalty is::

        penalty = penalty_weight * Σ_p ((cur_p - base_p) / |base_p|)^2 * score_scale

    over parameters ``p`` not in ``exclude``.

    Parameter set: ``param_names`` (e.g. the optimization parameter-space keys)
    when provided; otherwise every numeric scalar key shared by both configs,
    minus ``PARAM_PENALTY_DENYLIST``. Non-numeric / missing keys are skipped.
    """
    exclude = set(exclude or ())

    if param_names:
        keys = [k for k in param_names if k not in exclude]
    else:
        denylist = set(PARAM_PENALTY_DENYLIST)
        keys = [
            k for k in current_cfg
            if k in baseline_cfg and k not in exclude and k not in denylist
            and _is_number(current_cfg.get(k)) and _is_number(baseline_cfg.get(k))
        ]

    per_param: Dict[str, Dict[str, float]] = {}
    sq_terms: List[float] = []
    for k in keys:
        if k not in baseline_cfg or k not in current_cfg:
            continue
        bv, cv = baseline_cfg[k], current_cfg[k]
        if not (_is_number(bv) and _is_number(cv)):
            continue
        rel = (float(cv) - float(bv)) / max(abs(float(bv)), eps)
        sq = float(rel ** 2)
        per_param[k] = {
            'baseline': float(bv),
            'current': float(cv),
            'rel_diff': float(rel),
            'sq_rel_diff': sq,
        }
        sq_terms.append(sq)

    sum_sq = float(np.sum(sq_terms)) if sq_terms else 0.0
    penalty_score = float(penalty_weight) * sum_sq * float(score_scale)

    return {
        'per_param': per_param,
        'n_params_compared': len(sq_terms),
        'excluded_params': sorted(exclude),
        'sum_sq_rel_diff': sum_sq,
        'penalty_weight': float(penalty_weight),
        'penalty_score': penalty_score,
    }


# =============================================================================
# PLOTTING (bar chart of sim vs exp ratios — mirrors plot_drug_ratio_histograms)
# =============================================================================

def plot_drug_ratio_comparison(
    sim_ratios: Dict[str, float],
    exp_ratios: Dict[str, float],
    save_path: str,
    drug_name: str = "drug",
    fitness: Optional[float] = None,
) -> None:
    """Grouped horizontal bar chart: simulated vs experimental post/pre ratio
    per feature, with a reference line at 1.0 (no change)."""
    feats = [f for f in RATIO_FEATURES if f in sim_ratios and f in exp_ratios]
    if not feats:
        logger.warning("No common features to plot.")
        return

    sim_vals = [sim_ratios[f] for f in feats]
    exp_vals = [exp_ratios[f] for f in feats]
    y = np.arange(len(feats))[::-1]
    h = 0.38

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.barh(y + h / 2, sim_vals, height=h, color="#c0392b", label="simulated (post/pre)")
    ax.barh(y - h / 2, exp_vals, height=h, color="#2471a3", label="experimental (post/pre)")
    ax.axvline(1.0, color="k", lw=1.2, ls="--", label="no change (1.0)")
    ax.set_yticks(y)
    ax.set_yticklabels([f.replace("_ratio", "") for f in feats], fontsize=9)
    ax.set_xlabel("post / pre")
    title = f"Simulated vs experimental drug-response ratios — {drug_name}"
    if fitness is not None:
        title = f"Fitness={fitness:.2f} | " + title
    ax.set_title(title, fontsize=12)
    for yi, v in zip(y + h / 2, sim_vals):
        ax.text(v, yi, f" {v:.2f}", va="center", ha="left", fontsize=7, color="#c0392b")
    for yi, v in zip(y - h / 2, exp_vals):
        ax.text(v, yi, f" {v:.2f}", va="center", ha="left", fontsize=7, color="#2471a3")
    xmax = max(2.0, max(sim_vals + exp_vals) * 1.15)
    ax.set_xlim(0, xmax)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    logger.info("Saved drug-ratio comparison plot: %s", save_path)
    plt.close(fig)


def plot_drug_network_comparison(
    baseline_details: Dict,
    post_details: Dict,
    classification: Dict[int, str],
    save_path: str,
    drug_name: str = "drug",
    fitness: Optional[float] = None,
) -> None:
    """Baseline (pre-drug) vs post-drug simulated networks, side by side.

    Four stacked panels (shared time axis):
        Row 1: baseline raster (E/I coloured) + burst overlays
        Row 2: baseline network-activity signal + burst overlays
        Row 3: post-drug raster + burst overlays
        Row 4: post-drug network-activity signal + burst overlays

    Reuses the raster / network-signal / burst-overlay helpers from
    fitnessFunc_v2 so the panels match the standard fitness plots. Full-window
    plus 60 s and 30 s zoomed copies are written (same convention as v2).
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    ax_b_raster, ax_b_net, ax_p_raster, ax_p_net = axes

    # cell_types maps unit -> 'excitatory'/'inhibitory' (baseline-derived labels,
    # applied to both sides so colours are consistent).
    cell_types = dict(classification)

    title_prefix = f"Fitness={fitness:.2f} | " if fitness is not None else ""

    # --- Baseline ---
    _plot_raster(ax_b_raster, baseline_details.get('spike_data', {}),
                 title=f"{title_prefix}Baseline (pre-drug) sim raster — {drug_name}",
                 cell_types=cell_types)
    _overlay_bursts(ax_b_raster, baseline_details.get('burst_events', {}))
    b_pd = baseline_details.get('plot_data', {})
    if b_pd:
        _plot_network_signal(ax_b_net, b_pd, title="Baseline network activity")
        _overlay_bursts(ax_b_net, baseline_details.get('burst_events', {}))
    else:
        ax_b_net.text(0.5, 0.5, "No baseline burst data",
                      transform=ax_b_net.transAxes, ha='center')

    # --- Post-drug ---
    _plot_raster(ax_p_raster, post_details.get('spike_data', {}),
                 title=f"Post-drug sim raster — {drug_name}",
                 cell_types=cell_types)
    _overlay_bursts(ax_p_raster, post_details.get('burst_events', {}))
    p_pd = post_details.get('plot_data', {})
    if p_pd:
        _plot_network_signal(ax_p_net, p_pd, title="Post-drug network activity")
        _overlay_bursts(ax_p_net, post_details.get('burst_events', {}))
    else:
        ax_p_net.text(0.5, 0.5, "No post-drug burst data",
                      transform=ax_p_net.transAxes, ha='center')

    ax_p_net.set_xlabel("Time (s)")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.08)

    fig.savefig(save_path, dpi=200)
    logger.info("Saved drug network-comparison plot: %s", save_path)

    for xlim, suffix in ((60, '_60s'), (30, '_30s')):
        for ax in axes:
            ax.set_xlim(0, xlim)
        zoom_path = save_path.replace('.png', f'{suffix}.png').replace('.svg', f'{suffix}.svg')
        fig.savefig(zoom_path, dpi=200)

    plt.close(fig)


# =============================================================================
# DATA LOADING HELPERS
# =============================================================================

def _load_sim_pkl_units(
    pkl_path: str, recording_duration: Optional[float], min_time: Optional[float]
) -> Tuple[Dict[int, np.ndarray], Dict, float, Dict]:
    """Load a NetPyNE .pkl and return
    ({gid: spike_times_s}, popData, duration_s, simCfg_dict).

    Mutates the global netpyne ``sim`` object — call this for the baseline
    *after* the current candidate's simData dict has already been copied out.
    """
    if sim is None:
        raise ImportError("netpyne not available")
    sim.load(pkl_path)
    sim_data = sim.allSimData.todict().copy()
    pop_data = sim.net.allPops.copy()
    cfg_dict = sim.cfg.todict().copy() if hasattr(sim, 'cfg') and sim.cfg is not None else {}
    if recording_duration is None:
        recording_duration = float(getattr(sim.cfg, 'duration', 1000.0)) / 1000.0
        if min_time:
            recording_duration -= min_time
    feats = extract_simulated_features(sim_data, recording_duration, min_time=min_time)
    return feats['spike_data'], pop_data, recording_duration, cfg_dict


# =============================================================================
# MAIN FITNESS FUNCTION
# =============================================================================

def fitnessFunc_v2_drug(
    simulated_data_path: str = None,
    baseline_data_path: str = None,
    reference_data_path: str = None,
    drug_name: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
    config: Optional[Dict] = None,
    **kwargs,
) -> float:
    """Drug-response fitness function.

    Parameters
    ----------
    simulated_data_path : str or dict, optional
        The *current* (post-drug) candidate sim. In a batch run this is loaded
        from the call stack instead (set kwargs['batching']=True) — exactly like
        fitnessFunc_v2.
    baseline_data_path : str, optional
        The *previous* (baseline) sim .pkl. Fixed across the whole optimisation
        ("already set, no change"). May also be passed as kwargs['baseline_data_path'].
    reference_data_path : str, optional
        Baseline experimental target h5 holding /drug_effects/<drug_name>/ ratios
        (written by _scripts/extract_drug_effects.py). May be passed via kwargs.
    drug_name : str, optional
        Which /drug_effects/<name>/ group to score against. Defaults to
        config['drug_name'] / kwargs['drug_name'] / the first group found.

    Returns
    -------
    fitness : float
        Lower is better. Also writes <candidate>_drug_fitness.json.
    """
    time_start = time.time()

    if config is None:
        config = DEFAULT_CONFIG.copy()
    max_score = float(config.get('max_fitness', 30000.0))
    score_scale = float(config.get('score_scale', 100.0))

    fit_schema = kwargs.get('fit_schema', None) or DEFAULT_FIT_SCHEMA

    try:
        # ----------------------------------------------------------------
        # 1. Resolve inputs
        # ----------------------------------------------------------------
        if reference_data_path is None:
            reference_data_path = kwargs.get('reference_data_path')
        if baseline_data_path is None:
            baseline_data_path = kwargs.get('baseline_data_path')
        if drug_name is None:
            drug_name = kwargs.get('drug_name', config.get('drug_name'))

        if reference_data_path is None:
            logger.error("No reference_data_path (experimental drug target h5) provided")
            return max_score
        if baseline_data_path is None:
            logger.error("No baseline_data_path (previous/baseline sim) provided")
            return max_score

        # ----------------------------------------------------------------
        # 2. Load the CURRENT (post-drug) candidate sim
        # ----------------------------------------------------------------
        current_sim_data = None
        current_pop_data = kwargs.get('popData')
        current_cfg = {}
        if kwargs.get('batching', False):
            kwargs = get_sim_data_from_call_stack(kwargs)
            current_sim_data = kwargs.get('simData')
            current_pop_data = kwargs.get('popData', current_pop_data)
            current_cfg = kwargs.get('simCfg', {}) or {}
        elif kwargs.get('sim_data_path'):
            kwargs = get_sim_data_from_pkl(kwargs)
            current_sim_data = kwargs.get('simData')
            current_pop_data = kwargs.get('popData', current_pop_data)
            current_cfg = kwargs.get('simCfg', {}) or {}
        elif simulated_data_path is not None:
            if isinstance(simulated_data_path, dict):
                current_sim_data = simulated_data_path
            elif isinstance(simulated_data_path, str) and simulated_data_path.endswith('.pkl'):
                if sim is None:
                    raise ImportError("netpyne not available to load .pkl")
                sim.load(simulated_data_path)
                current_sim_data = sim.allSimData.todict().copy()
                current_pop_data = sim.net.allPops.copy()
                current_cfg = sim.cfg.todict().copy() if getattr(sim, 'cfg', None) is not None else {}
            else:
                current_sim_data = np.load(simulated_data_path, allow_pickle=True)

        if current_sim_data is None:
            logger.warning("No current simulated data")
            return max_score

        # Comparison window T (seconds): cool-down-aware, consistent for both sims.
        network_cool_down = kwargs.get('network_cool_down', 0.0)
        min_analysis_time = network_cool_down if network_cool_down > 0 else None
        recording_duration = kwargs.get('recording_duration')
        if recording_duration is None and isinstance(kwargs.get('simCfg'), dict):
            recording_duration = float(kwargs['simCfg'].get('duration', 0.0)) / 1000.0 or None
            if recording_duration and min_analysis_time:
                recording_duration -= min_analysis_time

        current_features = extract_simulated_features(
            current_sim_data, recording_duration, min_time=min_analysis_time,
        )
        if current_features['n_units'] == 0:
            logger.warning("No spikes in current (post-drug) simulation")
            return max_score
        current_units = current_features['spike_data']
        if recording_duration is None:
            recording_duration = current_features['recording_duration']

        # ----------------------------------------------------------------
        # 3. Load the BASELINE (previous) sim — fixed reference
        #    (done AFTER copying current data out of the global sim object)
        # ----------------------------------------------------------------
        baseline_units, baseline_pop_data, _bdur, baseline_cfg = _load_sim_pkl_units(
            baseline_data_path, recording_duration, min_analysis_time,
        )
        if not baseline_units:
            logger.warning("No spikes in baseline simulation")
            return max_score

        T = float(recording_duration)
        logger.info("Drug fitness: T=%.2f s | baseline units=%d | current units=%d",
                    T, len(baseline_units), len(current_units))

        # ----------------------------------------------------------------
        # 4. E/I classification (top-quantile on BASELINE, matching experiment)
        # ----------------------------------------------------------------
        use_true_ei = kwargs.get('use_true_ei', False)
        if use_true_ei and baseline_pop_data:
            classification = classify_units_from_popdata(baseline_pop_data)
            ei_threshold = float('nan')
            logger.info("E/I from baseline popData (ground truth): %d labelled units",
                        len(classification))
        else:
            classification, ei_threshold = classify_units_top_quantile(
                baseline_units, T, quantile=config.get('inhib_quantile', INHIB_QUANTILE),
            )
            n_inh = sum(1 for c in classification.values() if c == 'inhibitory')
            logger.info("E/I from baseline top-%.0f%% firing rate (thr=%.4f Hz): %d inh / %d exc",
                        (1.0 - config.get('inhib_quantile', INHIB_QUANTILE)) * 100,
                        ei_threshold, n_inh, len(classification) - n_inh)

        # ----------------------------------------------------------------
        # 5. Per-side features -> simulated post/pre ratios
        # ----------------------------------------------------------------
        want_plot = bool(kwargs.get('plot_sim', False))
        pre_features, pre_details = compute_sim_side_features(
            baseline_units, T, classification, "baseline", return_details=True)
        post_features, post_details = compute_sim_side_features(
            current_units, T, classification, "post-drug", return_details=True)
        sim_ratios = compute_sim_ratios(pre_features, post_features)

        # ----------------------------------------------------------------
        # 6. Experimental drug ratios + scoring
        # ----------------------------------------------------------------
        exp_ratios, resolved_drug = load_experimental_drug_ratios(reference_data_path, drug_name)
        if not exp_ratios:
            logger.error("No experimental drug ratios loaded")
            return max_score

        score = score_drug_ratios(
            sim_ratios, exp_ratios, fit_schema=fit_schema,
            score_scale=score_scale, max_score=max_score,
        )
        ratio_score = score['total_score']

        # ----------------------------------------------------------------
        # 6b. Config-drift penalty: the drug may only change the GABA inhibitory
        #     weights — everything else must match the baseline network.
        # ----------------------------------------------------------------
        # Explicit override (kwargs/config) wins; otherwise pick the params this
        # drug is allowed to change based on its name:
        #   bicuculline -> GABA weights ;  ap5_nbqx -> AMPA + NMDA weights.
        exclude_params = kwargs.get('exclude_params', config.get('exclude_params'))
        if not exclude_params:
            exclude_params = exclude_params_for_drug(resolved_drug)
        logger.info("Drift-penalty excluding (drug-allowed) params for '%s': %s",
                    resolved_drug, list(exclude_params))
        penalty_weight = kwargs.get('config_penalty_weight',
                                    config.get('config_penalty_weight', 1.0))
        penalty_weight = 1.0 if penalty_weight is None else float(penalty_weight)
        # The drift penalty is NOT multiplied by the ratio score_scale (100).
        # It uses its own scale (default 1.0) so penalty = weight * Σ rel_diff^2.
        penalty_scale = kwargs.get('config_penalty_scale',
                                   config.get('config_penalty_scale', 1.0))
        penalty_scale = 1.0 if penalty_scale is None else float(penalty_scale)
        param_names = kwargs.get('param_names') or None

        penalty = compute_config_penalty(
            baseline_cfg, current_cfg,
            param_names=param_names,
            exclude=tuple(exclude_params),
            penalty_weight=penalty_weight,
            score_scale=penalty_scale,
            eps=EPS,
        )
        penalty_score = penalty['penalty_score']
        logger.info("Config-drift penalty = %.4f over %d params (excl %s)",
                    penalty_score, penalty['n_params_compared'], penalty['excluded_params'])

        fitness = min(float(ratio_score) + float(penalty_score), max_score)
        logger.info("Drug-response fitness (%s) = %.4f (ratios=%.4f + penalty=%.4f)",
                    resolved_drug, fitness, ratio_score, penalty_score)

        # ----------------------------------------------------------------
        # 7. Build + save result
        # ----------------------------------------------------------------
        result = {
            'fitness': float(fitness),
            'fit': float(fitness),
            'drug_name': resolved_drug,
            'recording_duration': T,
            'ei_threshold_firing_rate_hz': float(ei_threshold) if np.isfinite(ei_threshold) else None,
            'ratio_score': float(ratio_score),
            'config_penalty_score': float(penalty_score),
            'sim_ratios': {k: float(v) for k, v in sim_ratios.items()},
            'exp_ratios': {k: float(v) for k, v in exp_ratios.items()},
            'fitness_dict': {
                'drug_ratios': {'fit': float(ratio_score), **score},
                'config_drift_penalty': {'fit': float(penalty_score), **penalty},
            },
            'baseline_features': {k: float(v) for k, v in pre_features.items()},
            'post_drug_features': {k: float(v) for k, v in post_features.items()},
            'paths': {
                'baseline_sim': baseline_data_path,
                'reference_h5': reference_data_path,
            },
        }

        candidate_path = kwargs.get('candidate_path')
        sim_data_path_kw = kwargs.get('sim_data_path')
        if candidate_path is None and isinstance(sim_data_path_kw, str):
            candidate_path = sim_data_path_kw.replace('_data.pkl', '').replace('.npy', '')

        fitness_save_path = kwargs.get('fitness_save_path')
        if fitness_save_path is None and candidate_path:
            fitness_save_path = f'{candidate_path}_drug_fitness.json'

        if fitness_save_path:
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
            logger.info("Saved drug fitness to %s", fitness_save_path)

        # ----------------------------------------------------------------
        # 8. Optional plots
        # ----------------------------------------------------------------
        if want_plot and candidate_path:
            # (a) sim-vs-exp ratio bar chart
            try:
                plot_drug_ratio_comparison(
                    sim_ratios, exp_ratios,
                    save_path=f'{candidate_path}_drug_ratio_plot.png',
                    drug_name=resolved_drug, fitness=fitness,
                )
            except Exception as e:
                logger.warning("Drug-ratio plotting failed: %s", e)

            # (b) baseline-vs-post-drug network comparison (rasters + activity)
            try:
                plot_drug_network_comparison(
                    pre_details, post_details, classification,
                    save_path=f'{candidate_path}_drug_network_comparison.png',
                    drug_name=resolved_drug, fitness=fitness,
                )
            except Exception as e:
                logger.warning("Drug network-comparison plotting failed: %s", e)

        logger.info("Elapsed: %.2fs", time.time() - time_start)
        return float(fitness)

    except Exception as e:
        logger.error("Error computing drug fitness: %s", e)
        traceback.print_exc()
        return max_score


# =============================================================================
# CONVENIENCE / CLI
# =============================================================================

def compute_drug_fitness(
    sim_data: Any,
    baseline_data_path: str,
    exp_h5_path: str,
    drug_name: Optional[str] = None,
    weights: Optional[Dict] = None,
) -> Tuple[float, Dict]:
    """Convenience wrapper returning (fitness_value, {})."""
    fitness = fitnessFunc_v2_drug(
        simulated_data_path=sim_data,
        baseline_data_path=baseline_data_path,
        reference_data_path=exp_h5_path,
        drug_name=drug_name,
        weights=weights,
    )
    return fitness, {}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fitness Function v2 (drug) — standalone mode")
    parser.add_argument("--sim", required=True, help="Current (post-drug) sim .pkl/.npy")
    parser.add_argument("--baseline", required=True, help="Baseline (previous) sim .pkl")
    parser.add_argument("--ref", required=True, help="Experimental target h5 with /drug_effects/")
    parser.add_argument("--drug_name", default=None, help="Drug group key (default: first found)")
    parser.add_argument("--plot", action="store_true", help="Save sim-vs-exp ratio bar chart")
    parser.add_argument("--output", default=None, help="Output drug_fitness.json path")
    args = parser.parse_args()

    candidate_path = args.sim.replace('_data.pkl', '').replace('.npy', '')
    fitness_save = args.output or f"{candidate_path}_drug_fitness.json"

    fitness = fitnessFunc_v2_drug(
        simulated_data_path=args.sim,
        baseline_data_path=args.baseline,
        reference_data_path=args.ref,
        drug_name=args.drug_name,
        plot_sim=args.plot,
        candidate_path=candidate_path,
        fitness_save_path=fitness_save,
    )

    print(f"\nDrug-response fitness: {fitness:.4f}")
    print(f"Saved:                 {fitness_save}")
