"""Compare experimental vs simulated drug-response change vectors.

Background
----------
Given a drug (e.g. bicuculline), the experimental side has a "change vector":
the post/pre ratio of each population-level feature, measured on paired raw
Maxwell recordings (see ``plot_drug_ratio_histograms.py`` /
``extract_drug_effects.py``). The simulated side has the analogous change
vector: the same features computed on two NetPyNE sims of the *same* candidate
parameter set — one baseline-cfg sim, one drug-cfg sim.

This script builds both vectors and shows how far the simulated response
matches the experimental one, both as scalar ratios *and* as full per-unit
distributions (so you can see *how* the population redistributes under the
drug, not just the population mean).

Inputs
------
Experimental (raw extraction, same chain as plot_drug_ratio_histograms.py):
    --baseline_raw  baseline Maxwell .h5
    --drug_raw      post-drug Maxwell .h5
    --well          stream id (default well000)
    --baseline_target  optional h5 to pull T_target_s from

Simulated:
    --baseline_sim_fitness  trial_<k>_fitness.json from a baseline-cfg sim
    --drug_sim_fitness      trial_<k>_fitness.json from a drug-cfg sim
    --baseline_sim_pkl      trial_<k>_data.pkl from baseline-cfg sim
                            (optional — enables per-unit histogram pages)
    --drug_sim_pkl          trial_<k>_data.pkl from drug-cfg sim
                            (optional — enables per-unit histogram pages)

Output
------
A multi-page PDF + sidecar JSON:
  Page 1  Per-feature bar chart of scalar ratios (capped at --max_ratio_display
          so outliers like a 1.5e8 superburst blowup don't dominate the axis).
  Page 2  Per-unit firing-rate distributions, pre vs post, exp (top) and sim
          (bottom). Log-x; capped at --max_rate_display Hz.
  Page 3  Per-unit ratio histograms (post/pre), exp (top) and sim (bottom).
          log2 axis clipped to ±--max_log2_clip.
  Page 4  Scalar change-vector difference log2(sim)-log2(exp) per feature,
          clipped to ±--max_log2_clip. RMS is computed over in-range features.
  Page 5  sim vs exp ratio scatter on log2 axes with y=x.

The histogram pages are skipped if you don't pass both --baseline_sim_pkl and
--drug_sim_pkl.

Usage
-----
  module load conda && conda activate preshifter
  HDF5_PLUGIN_PATH=/global/homes/k/ktub1999/hdf5_plugin_path_maxwell \\
  python RBS_network_models/_scripts/compare_drug_change_vectors.py \\
    --baseline_raw experimental_data/CDKL5_02.h5 \\
    --drug_raw     experimental_data/Immediately_after_drug_11.h5 \\
    --baseline_target RBS_network_models/processed_experimental_targets/CDKL5_002_well000.h5 \\
    --baseline_sim_fitness <path>/gen_X/trial_Y_fitness.json \\
    --drug_sim_fitness     <path>/gen_X/trial_Y_fitness.json \\
    --baseline_sim_pkl     <path>/gen_X/trial_Y_data.pkl \\
    --drug_sim_pkl         <path>/gen_X/trial_Y_data.pkl \\
    --drug_name bicuculline \\
    --output compare_change_vectors_bicuculline.pdf
"""

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from extract_drug_effects import (  # noqa: E402
    EPS,
    compute_ratios,
    get_T_from_baseline_h5,
    load_from_raw_recordings,
    _per_channel_firing_rates_hz,
    _truncate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compare_drug_change_vectors")


# Features we can recover from a trial_*_fitness.json. Mapped onto the same
# ratio names used by extract_drug_effects so the two vectors line up.
SIM_RATIO_FEATURES = (
    "pop_FR_ratio",
    "exc_firing_rate_ratio",
    "inh_firing_rate_ratio",
    "burstlet_rate_ratio",
    "network_burst_rate_ratio",
    "superburst_rate_ratio",
    "pre_burstlet_rate_ratio",
)


def _safe_ratio(num: float, denom: float) -> float:
    return float(num) / max(float(denom), EPS)


# ---------- Simulated-side feature extraction -----------------------------

def _sim_side_features(fitness_json_path: Path) -> dict:
    """Per-side population features from a single trial_*_fitness.json."""
    with open(fitness_json_path) as f:
        d = json.load(f)
    sim_metrics = d.get("simulated_metrics", {})
    fd = d.get("fitness_dict", {})
    um = fd.get("unit_metrics", {})

    T = float(sim_metrics.get("recording_duration", 0.0))
    if T <= 0.0:
        raise RuntimeError(f"recording_duration missing/zero in {fitness_json_path}")

    def _bcount(level: str) -> float:
        return float(fd.get(level, {}).get("n_sim", 0))

    return {
        "T_seconds":               T,
        "pop_FR_hz":               float(sim_metrics.get("mean_firing_rate", 0.0)),
        "exc_firing_rate_hz":      float(um.get("sim_mean_firing_rate_exc", 0.0)),
        "inh_firing_rate_hz":      float(um.get("sim_mean_firing_rate_inh", 0.0)),
        "burstlet_rate_hz":        _bcount("burstlets") / T,
        "network_burst_rate_hz":   _bcount("network_bursts") / T,
        "superburst_rate_hz":      _bcount("superbursts") / T,
        "pre_burstlet_rate_hz":    _bcount("pre_burstlets") / T,
        "n_units":                 int(sim_metrics.get("n_units", 0)),
        "total_spikes":            int(sim_metrics.get("total_spikes", 0)),
        "_source":                 str(fitness_json_path),
    }


def compute_sim_ratios(
    baseline_json: Path, drug_json: Path,
) -> Tuple[Dict[str, float], dict]:
    pre = _sim_side_features(baseline_json)
    post = _sim_side_features(drug_json)
    logger.info(
        "[sim:pre]  pop_FR=%.4f Hz  exc_FR=%.4f Hz  inh_FR=%.4f Hz  T=%.2fs  (%d units)",
        pre["pop_FR_hz"], pre["exc_firing_rate_hz"], pre["inh_firing_rate_hz"],
        pre["T_seconds"], pre["n_units"],
    )
    logger.info(
        "[sim:post] pop_FR=%.4f Hz  exc_FR=%.4f Hz  inh_FR=%.4f Hz  T=%.2fs  (%d units)",
        post["pop_FR_hz"], post["exc_firing_rate_hz"], post["inh_firing_rate_hz"],
        post["T_seconds"], post["n_units"],
    )
    ratios = {
        "pop_FR_ratio":              _safe_ratio(post["pop_FR_hz"], pre["pop_FR_hz"]),
        "exc_firing_rate_ratio":     _safe_ratio(post["exc_firing_rate_hz"], pre["exc_firing_rate_hz"]),
        "inh_firing_rate_ratio":     _safe_ratio(post["inh_firing_rate_hz"], pre["inh_firing_rate_hz"]),
        "burstlet_rate_ratio":       _safe_ratio(post["burstlet_rate_hz"], pre["burstlet_rate_hz"]),
        "network_burst_rate_ratio":  _safe_ratio(post["network_burst_rate_hz"], pre["network_burst_rate_hz"]),
        "superburst_rate_ratio":     _safe_ratio(post["superburst_rate_hz"], pre["superburst_rate_hz"]),
        "pre_burstlet_rate_ratio":   _safe_ratio(post["pre_burstlet_rate_hz"], pre["pre_burstlet_rate_hz"]),
    }
    return ratios, {"pre": pre, "post": post}


def _load_sim_per_unit_rates(pkl_path: Path) -> Tuple[Dict[int, float], float]:
    """Load NetPyNE sim pkl → {gid: firing_rate_Hz}, T_seconds.

    spkt is in ms; duration is taken from simData['simDuration']/simConfig
    (also in ms). Units that never fired aren't enumerated here — same
    convention plot_drug_ratio_histograms.py uses (the common-units
    intersection later drops silent-on-both-sides anyway).
    """
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    sd = d.get("simData", d.get("allSimData", d))
    spkt = np.asarray(sd.get("spkt", sd.get("spkts", [])), dtype=float)  # ms
    spkid = np.asarray(sd.get("spkid", sd.get("spkids", [])), dtype=int)

    duration_ms = sd.get("simDuration") or sd.get("duration")
    if duration_ms is None and isinstance(d.get("simConfig"), dict):
        duration_ms = d["simConfig"].get("duration")
    if duration_ms is None and isinstance(d.get("cfg"), dict):
        duration_ms = d["cfg"].get("duration")
    if duration_ms is None:
        duration_ms = float(spkt.max()) if spkt.size else 1000.0
    T = float(duration_ms) / 1000.0

    rates: Dict[int, float] = {}
    if spkt.size and spkid.size:
        gids, counts = np.unique(spkid, return_counts=True)
        for gid, c in zip(gids, counts):
            rates[int(gid)] = float(c) / T
    return rates, T


def _common_unit_stats(
    pre_rates: Dict, post_rates: Dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Per-unit stats over the id intersection (mirrors per_channel_stats
    in plot_drug_ratio_histograms.py). Returns
    (pre_arr, post_arr, both_active_ratios, counts)."""
    common = sorted(set(pre_rates) & set(post_rates))
    pre_all = np.array([pre_rates[c] for c in common])
    post_all = np.array([post_rates[c] for c in common])
    ratios = []
    n_sil = n_act = n_silboth = 0
    for c in common:
        pr, po = pre_rates[c], post_rates[c]
        if pr > 0 and po > 0:
            ratios.append(po / pr)
        elif pr > 0:
            n_sil += 1
        elif po > 0:
            n_act += 1
        else:
            n_silboth += 1
    counts = {
        "total_common": len(common),
        "both_active": len(ratios),
        "silenced": n_sil,
        "activated": n_act,
        "silent_both": n_silboth,
    }
    return pre_all, post_all, np.asarray(ratios), counts


# ---------- Plotting -------------------------------------------------------

def _short(name: str) -> str:
    return name.replace("_ratio", "")


def _fmt_ratio_power(p: int) -> str:
    if p == 0:
        return "1x"
    if p > 0:
        return f"{2 ** p}x"
    return f"1/{2 ** (-p)}"


def _summary_bars_page(
    pdf: PdfPages,
    drug_name: str,
    feats: list,
    exp_vals: np.ndarray,
    sim_vals: np.ndarray,
    exp_meta: dict,
    sim_meta: dict,
    *,
    max_ratio_display: float,
) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(
        f"Drug-response ratios (post / pre) — {drug_name}\nExperimental vs Simulated",
        fontsize=14, fontweight="bold",
    )

    ax_txt = fig.add_axes([0.04, 0.08, 0.30, 0.80])
    ax_txt.axis("off")
    lines = [
        f"drug              : {drug_name}",
        "",
        "[experimental]",
        f"  source          : {exp_meta.get('source', '?')}",
        f"  T window (s)    : {exp_meta.get('T_seconds', float('nan')):.2f}",
        f"  pre units       : {exp_meta.get('n_pre_units', '?')}",
        f"  post units      : {exp_meta.get('n_post_units', '?')}",
        f"  baseline_raw    : {os.path.basename(str(exp_meta.get('baseline_raw', '')))}",
        f"  drug_raw        : {os.path.basename(str(exp_meta.get('drug_raw', '')))}",
        "",
        "[simulated]",
        f"  baseline T (s)  : {sim_meta['pre']['T_seconds']:.2f}",
        f"  drug T (s)      : {sim_meta['post']['T_seconds']:.2f}",
        f"  pre n_units     : {sim_meta['pre']['n_units']}",
        f"  post n_units    : {sim_meta['post']['n_units']}",
        f"  baseline json   : {os.path.basename(sim_meta['pre']['_source'])}",
        f"  drug json       : {os.path.basename(sim_meta['post']['_source'])}",
        "",
        f"axis cap          : {max_ratio_display:g}x",
    ]
    ax_txt.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
                family="monospace", fontsize=9, transform=ax_txt.transAxes)

    ax = fig.add_axes([0.40, 0.08, 0.56, 0.80])
    y = np.arange(len(feats))[::-1]
    h = 0.38
    exp_clip = np.minimum(exp_vals, max_ratio_display)
    sim_clip = np.minimum(sim_vals, max_ratio_display)
    ax.barh(y + h / 2, exp_clip, height=h, color="#2471a3", label="experimental")
    ax.barh(y - h / 2, sim_clip, height=h, color="#c0392b", label="simulated")
    ax.axvline(1.0, color="k", lw=1.2, ls="--", label="no change (1.0)")
    ax.set_yticks(y)
    ax.set_yticklabels([_short(f) for f in feats], fontsize=9)
    ax.set_xlabel(f"post / pre  (clipped at {max_ratio_display:g}x)")
    ax.set_title("Per-feature ratio: experimental vs simulated", fontsize=11)

    def _annotate(yi, v_clip, v_raw, color):
        if v_raw > max_ratio_display:
            txt = f" ▶ {v_raw:.2g} (clipped)"
        else:
            txt = f" {v_raw:.2f}"
        ax.text(v_clip, yi, txt, va="center", ha="left", fontsize=7, color=color)

    for yi, vc, vr in zip(y + h / 2, exp_clip, exp_vals):
        _annotate(yi, vc, vr, "#2471a3")
    for yi, vc, vr in zip(y - h / 2, sim_clip, sim_vals):
        _annotate(yi, vc, vr, "#c0392b")
    ax.set_xlim(0, max_ratio_display * 1.18)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    pdf.savefig(fig)
    plt.close(fig)


def _per_unit_scatter_page(
    pdf: PdfPages,
    drug_name: str,
    exp_pre: np.ndarray, exp_post: np.ndarray,
    sim_pre: np.ndarray, sim_post: np.ndarray,
    *, max_rate_display: float, floor: float = 0.01,
) -> None:
    """One point per unit: x=pre rate, y=post rate. y=x = no change.

    Silenced units (post=0) collapse to the bottom edge (drawn as ▼ at y=floor);
    activated units (pre=0) collapse to the left edge (drawn as ▲ at x=floor).
    Log–log axes; rates above max_rate_display are clipped to the right/top edge.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        f"Per-unit firing rate, pre vs post — {drug_name}",
        fontsize=13, fontweight="bold",
    )

    def _draw(ax, pre, post, label):
        pre = np.asarray(pre, dtype=float)
        post = np.asarray(post, dtype=float)
        if pre.size == 0:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"{label}: no units",
                    transform=ax.transAxes, ha="center", va="center")
            return
        pre_zero = pre <= 0
        post_zero = post <= 0
        both = ~pre_zero & ~post_zero
        silenced = ~pre_zero & post_zero
        activated = pre_zero & ~post_zero

        pre_plot = np.clip(pre, floor, max_rate_display)
        post_plot = np.clip(post, floor, max_rate_display)

        if both.any():
            up = both & (post > pre)
            down = both & (post < pre)
            same = both & (post == pre)
            if up.any():
                ax.scatter(pre_plot[up], post_plot[up], s=14, c="#c0392b",
                           alpha=0.55, edgecolor="none",
                           label=f"increased (n={int(up.sum())})")
            if down.any():
                ax.scatter(pre_plot[down], post_plot[down], s=14, c="#2471a3",
                           alpha=0.55, edgecolor="none",
                           label=f"decreased (n={int(down.sum())})")
            if same.any():
                ax.scatter(pre_plot[same], post_plot[same], s=14, c="0.4",
                           alpha=0.55, edgecolor="none",
                           label=f"unchanged (n={int(same.sum())})")
        if silenced.any():
            ax.scatter(pre_plot[silenced], np.full(int(silenced.sum()), floor),
                       s=22, c="#7d3c98", marker="v", alpha=0.7,
                       label=f"silenced (post=0, n={int(silenced.sum())})")
        if activated.any():
            ax.scatter(np.full(int(activated.sum()), floor), post_plot[activated],
                       s=22, c="#117a65", marker="^", alpha=0.7,
                       label=f"activated (pre=0, n={int(activated.sum())})")

        ax.plot([floor, max_rate_display], [floor, max_rate_display],
                "k--", lw=1.0, label="y = x (no change)")
        ax.axhline(1.0, color="0.7", lw=0.6, ls=":")
        ax.axvline(1.0, color="0.7", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(floor, max_rate_display)
        ax.set_ylim(floor, max_rate_display)
        ax.set_xlabel(f"pre firing rate (Hz, clipped to [{floor:g}, {max_rate_display:g}])")
        ax.set_ylabel(f"post firing rate (Hz, clipped to [{floor:g}, {max_rate_display:g}])")
        ax.set_title(label, fontsize=11)
        ax.legend(loc="lower right", fontsize=7.5)
        ax.grid(alpha=0.3, which="both")

    _draw(axes[0], exp_pre, exp_post, "experimental (per-channel)")
    _draw(axes[1], sim_pre, sim_post, "simulated (per-cell)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def _distribution_overlay_page(
    pdf: PdfPages,
    drug_name: str,
    exp_pre: np.ndarray, exp_post: np.ndarray,
    sim_pre: np.ndarray, sim_post: np.ndarray,
    *, bins: int, max_rate_display: float,
) -> None:
    """Pre-vs-post per-unit firing-rate distributions, exp (top) and sim (bottom)."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
    fig.suptitle(
        f"Per-unit firing-rate distributions, pre vs post — {drug_name}",
        fontsize=13, fontweight="bold",
    )

    def _draw(ax, pre, post, label):
        pre = np.asarray(pre)
        post = np.asarray(post)
        if pre.size == 0 and post.size == 0:
            ax.text(0.5, 0.5, f"{label}: no units", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_axis_off()
            return
        pre_clip = pre[pre <= max_rate_display]
        post_clip = post[post <= max_rate_display]
        pos = np.concatenate([pre_clip[pre_clip > 0], post_clip[post_clip > 0]])
        if pos.size == 0:
            ax.text(0.5, 0.5, f"{label}: no positive rates after clipping",
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            return
        lo = max(float(pos.min()), 1e-3)
        hi = max(float(pos.max()), lo * 2)
        edges = np.logspace(np.log10(lo), np.log10(hi), bins + 1)
        ax.hist(pre_clip[pre_clip > 0], bins=edges, alpha=0.55, color="#2471a3",
                label=f"pre  median={np.median(pre):.3f} Hz  n={pre.size}")
        ax.hist(post_clip[post_clip > 0], bins=edges, alpha=0.55, color="#c0392b",
                label=f"post median={np.median(post):.3f} Hz  n={post.size}")
        n_clip_pre = int((pre > max_rate_display).sum())
        n_clip_post = int((post > max_rate_display).sum())
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_xlabel(
            f"firing rate (Hz, log; clipped at {max_rate_display:g} Hz, "
            f"excluded pre={n_clip_pre} post={n_clip_post})"
        )
        ax.set_ylabel("number of units")
        ax.set_title(label, fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)

    _draw(axes[0], exp_pre, exp_post, "experimental (per-channel)")
    _draw(axes[1], sim_pre, sim_post, "simulated (per-cell)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)


def _ratio_distribution_page(
    pdf: PdfPages,
    drug_name: str,
    exp_ratios_pu: np.ndarray,
    sim_ratios_pu: np.ndarray,
    *, bins: int, max_log2_clip: float,
    exp_counts: dict, sim_counts: dict,
) -> None:
    """Per-unit log2 ratio histograms, exp (top) and sim (bottom)."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
    fig.suptitle(
        f"Per-unit ratio histograms (post/pre) — {drug_name}",
        fontsize=13, fontweight="bold",
    )

    def _draw(ax, ratios, counts, label):
        ratios = np.asarray(ratios)
        if ratios.size == 0:
            ax.text(0.5, 0.5, f"{label}: no both-active units",
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            return
        log2r_raw = np.log2(ratios)
        lo, hi = np.percentile(log2r_raw, [1.0, 99.0])
        span = float(min(max_log2_clip, max(abs(lo), abs(hi), 2.0)))
        span = float(np.ceil(span))
        log2r = np.clip(log2r_raw, -span, span)
        edges = np.linspace(-span, span, bins + 1)
        ax.hist(log2r, bins=edges, color="0.6", alpha=0.6,
                label=f"all (n={log2r.size})")
        ax.axvline(0.0, color="k", lw=1.2, ls="--", label="no change (1x)")
        med = float(np.median(log2r_raw))
        if -span <= med <= span:
            ax.axvline(med, color="#c0392b", lw=1.5, ls=":",
                       label=f"median = {2.0 ** med:.3g}x")
        step = max(1, int(np.ceil(span / 6.0)))
        tick_pow = np.arange(-int(span), int(span) + 1, step)
        ax.set_xticks(tick_pow)
        ax.set_xticklabels([_fmt_ratio_power(int(p)) for p in tick_pow])
        ax.set_xlim(-span, span)
        ax.set_xlabel(
            f"post / pre per-unit ratio (log2; clipped to "
            f"[1/{2 ** int(span)}, {2 ** int(span)}x])"
        )
        ax.set_ylabel("number of units")
        frac_up = float(np.mean(ratios > 1.0))
        ax.text(0.02, 0.97,
                f"both-active: {counts.get('both_active', ratios.size)}\n"
                f"  {frac_up * 100:.0f}% increased / {(1 - frac_up) * 100:.0f}% decreased\n"
                f"silenced (pre>0, post=0): {counts.get('silenced', 0)}\n"
                f"activated (pre=0, post>0): {counts.get('activated', 0)}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
        ax.set_title(label, fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)

    _draw(axes[0], exp_ratios_pu, exp_counts, "experimental (per-channel)")
    _draw(axes[1], sim_ratios_pu, sim_counts, "simulated (per-cell)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)


def _difference_vector_page(
    pdf: PdfPages,
    drug_name: str,
    feats: list,
    exp_vals: np.ndarray,
    sim_vals: np.ndarray,
    *, max_log2_clip: float,
) -> None:
    diff = np.log2(np.maximum(sim_vals, EPS)) - np.log2(np.maximum(exp_vals, EPS))
    diff_clip = np.clip(diff, -max_log2_clip, max_log2_clip)

    fig, ax = plt.subplots(figsize=(11, 8.5))
    y = np.arange(len(feats))[::-1]
    colors = ["#c0392b" if v > 0 else "#2471a3" for v in diff_clip]
    ax.barh(y, diff_clip, color=colors, height=0.6)
    ax.axvline(0.0, color="k", lw=1.2, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([_short(f) for f in feats], fontsize=9)
    ax.set_xlabel(
        f"log2(sim ratio) - log2(exp ratio)  (clipped to ±{max_log2_clip:g})\n"
        "(>0: sim over-responded, <0: sim under-responded)"
    )
    ax.set_title(f"Change-vector difference (sim - exp, log2) — {drug_name}",
                 fontsize=13)
    for yi, vc, vr in zip(y, diff_clip, diff):
        mult = 2.0 ** vr
        if abs(vr) > max_log2_clip:
            txt = f" ▶ raw {vr:+.1f}  ({mult:.2g}x, clipped)"
        else:
            txt = f" {vr:+.2f}  ({mult:.2f}x)"
        ax.text(vc, yi, txt, va="center",
                ha="left" if vc >= 0 else "right", fontsize=8)
    ax.set_xlim(-max_log2_clip * 1.30, max_log2_clip * 1.30)

    in_range = np.abs(diff) <= max_log2_clip
    if in_range.any():
        rms_in = float(np.sqrt(np.mean(diff[in_range] ** 2)))
        n_in = int(in_range.sum())
    else:
        rms_in, n_in = float("nan"), 0
    ax.text(0.02, 0.97,
            f"features: {len(feats)}  ({n_in} in range, {len(feats) - n_in} clipped)\n"
            f"RMS log2 deviation (in-range only): {rms_in:.3f}  ({2.0 ** rms_in:.2f}x)",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.grid(axis="x", alpha=0.3)
    pdf.savefig(fig)
    plt.close(fig)


def _scatter_page(
    pdf: PdfPages,
    drug_name: str,
    feats: list,
    exp_vals: np.ndarray,
    sim_vals: np.ndarray,
    *, max_log2_clip: float,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    lo_v = 2.0 ** (-max_log2_clip)
    hi_v = 2.0 ** max_log2_clip
    e = np.clip(np.maximum(exp_vals, EPS), lo_v, hi_v)
    s = np.clip(np.maximum(sim_vals, EPS), lo_v, hi_v)
    clipped_mask = (exp_vals > hi_v) | (exp_vals < lo_v) | (sim_vals > hi_v) | (sim_vals < lo_v)
    ax.scatter(e[~clipped_mask], s[~clipped_mask], s=70, c="#2c3e50", zorder=3, label="in range")
    if clipped_mask.any():
        ax.scatter(e[clipped_mask], s[clipped_mask], s=70, c="#e67e22",
                   marker="x", zorder=3, label="clipped")
    for ei, si, name in zip(e, s, feats):
        ax.annotate(_short(name), (ei, si), xytext=(5, 5),
                    textcoords="offset points", fontsize=8)
    ax.plot([lo_v, hi_v], [lo_v, hi_v], "k--", lw=1.0, label="y = x (sim matches exp)")
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax.axvline(1.0, color="0.6", lw=0.8, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlim(lo_v, hi_v)
    ax.set_ylim(lo_v, hi_v)
    ax.set_xlabel(f"experimental ratio (post / pre)  [axis clipped to ±{max_log2_clip:g} log2]")
    ax.set_ylabel(f"simulated ratio (post / pre)  [axis clipped to ±{max_log2_clip:g} log2]")
    ax.set_title(f"Simulated vs experimental change — {drug_name}", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    pdf.savefig(fig)
    plt.close(fig)


# ---------- Main -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Experimental side
    parser.add_argument("--baseline_raw", required=True)
    parser.add_argument("--drug_raw", required=True)
    parser.add_argument("--well", default="well000")
    parser.add_argument("--baseline_target")
    parser.add_argument("--threshold_mad", type=float, default=5.0)
    parser.add_argument("--peak_sign", choices=("neg", "pos", "both"), default="neg")
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--T_seconds", type=float, default=None)
    # Simulated side (scalar features always required; pkls optional for histograms)
    parser.add_argument("--baseline_sim_fitness", required=True)
    parser.add_argument("--drug_sim_fitness", required=True)
    parser.add_argument("--baseline_sim_pkl",
                        help="trial_<k>_data.pkl baseline sim (enables per-unit histograms).")
    parser.add_argument("--drug_sim_pkl",
                        help="trial_<k>_data.pkl drug sim (enables per-unit histograms).")
    # Display caps
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--max_ratio_display", type=float, default=8.0,
                        help="Cap for scalar bar-chart x-axis (default 8.0).")
    parser.add_argument("--max_log2_clip", type=float, default=6.0,
                        help="Cap for log2 difference & per-unit ratio axes (default 6 = 64x).")
    parser.add_argument("--max_rate_display", type=float, default=30.0,
                        help="Cap for per-unit firing-rate axis in Hz (default 30 Hz).")
    parser.add_argument("--drug_name", default="drug")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    for p in (args.baseline_raw, args.drug_raw,
              args.baseline_sim_fitness, args.drug_sim_fitness):
        if not Path(p).exists():
            parser.error(f"File not found: {p}")
    for p in (args.baseline_sim_pkl, args.drug_sim_pkl):
        if p and not Path(p).exists():
            parser.error(f"File not found: {p}")
    have_pkls = bool(args.baseline_sim_pkl and args.drug_sim_pkl)
    if bool(args.baseline_sim_pkl) ^ bool(args.drug_sim_pkl):
        parser.error("Pass BOTH --baseline_sim_pkl and --drug_sim_pkl (or neither).")

    # --- experimental: raw extraction + per-channel firing rates ---
    pre_units, post_units, exp_source = load_from_raw_recordings(
        baseline_raw=Path(args.baseline_raw).resolve(),
        drug_raw=Path(args.drug_raw).resolve(),
        well=args.well, threshold_mad=args.threshold_mad,
        peak_sign=args.peak_sign, n_jobs=args.n_jobs,
    )
    T = args.T_seconds
    if T is None and args.baseline_target:
        T = get_T_from_baseline_h5(Path(args.baseline_target).resolve())
    if T is None:
        durations = []
        for u in (pre_units, post_units):
            if u:
                durations.append(max(t.max() for t in u.values() if len(t)))
        T = float(min(durations)) if durations else 0.0
        logger.warning("No T provided; defaulting to min recording duration: %.2f s", T)
    logger.info("Experimental window T = %.2f s", T)
    pre_clip = _truncate(pre_units, T)
    post_clip = _truncate(post_units, T)
    exp_ratios, exp_diag = compute_ratios(pre_clip, post_clip, T)
    exp_pre_rates = _per_channel_firing_rates_hz(pre_clip, T)
    exp_post_rates = _per_channel_firing_rates_hz(post_clip, T)
    exp_pre_arr, exp_post_arr, exp_ratios_pu, exp_counts = _common_unit_stats(
        exp_pre_rates, exp_post_rates,
    )
    logger.info("Experimental ratios:")
    for k, v in exp_ratios.items():
        logger.info("  %-30s = %.4f", k, v)
    logger.info("Experimental per-channel: %d both-active, %d silenced, %d activated, %d silent-both",
                exp_counts["both_active"], exp_counts["silenced"],
                exp_counts["activated"], exp_counts["silent_both"])

    # --- simulated scalar ratios ---
    sim_ratios, sim_diag = compute_sim_ratios(
        Path(args.baseline_sim_fitness).resolve(),
        Path(args.drug_sim_fitness).resolve(),
    )
    logger.info("Simulated ratios:")
    for k, v in sim_ratios.items():
        logger.info("  %-30s = %.4f", k, v)

    # --- simulated per-unit rates from pkls (if provided) ---
    sim_pre_arr = sim_post_arr = sim_ratios_pu = None
    sim_counts = None
    if have_pkls:
        sim_pre_rates, sim_T_pre = _load_sim_per_unit_rates(Path(args.baseline_sim_pkl).resolve())
        sim_post_rates, sim_T_post = _load_sim_per_unit_rates(Path(args.drug_sim_pkl).resolve())
        logger.info("[sim:pkl] baseline: %d active gids, T=%.2fs  drug: %d active gids, T=%.2fs",
                    len(sim_pre_rates), sim_T_pre, len(sim_post_rates), sim_T_post)
        sim_pre_arr, sim_post_arr, sim_ratios_pu, sim_counts = _common_unit_stats(
            sim_pre_rates, sim_post_rates,
        )
        logger.info("Simulated per-cell: %d both-active, %d silenced, %d activated, %d silent-both",
                    sim_counts["both_active"], sim_counts["silenced"],
                    sim_counts["activated"], sim_counts["silent_both"])

    feats = [f for f in SIM_RATIO_FEATURES if f in exp_ratios and f in sim_ratios]
    exp_vals = np.array([exp_ratios[f] for f in feats], dtype=float)
    sim_vals = np.array([sim_ratios[f] for f in feats], dtype=float)

    exp_meta = {
        "source": exp_source.get("source", "raw_recordings"),
        "baseline_raw": exp_source.get("baseline_raw", args.baseline_raw),
        "drug_raw": exp_source.get("drug_raw", args.drug_raw),
        "T_seconds": T,
        "n_pre_units": exp_diag.get("n_pre_units", "?"),
        "n_post_units": exp_diag.get("n_post_units", "?"),
    }

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        _summary_bars_page(pdf, args.drug_name, feats, exp_vals, sim_vals,
                           exp_meta, sim_diag,
                           max_ratio_display=args.max_ratio_display)
        if have_pkls:
            _per_unit_scatter_page(
                pdf, args.drug_name, exp_pre_arr, exp_post_arr,
                sim_pre_arr, sim_post_arr,
                max_rate_display=args.max_rate_display,
            )
            _distribution_overlay_page(
                pdf, args.drug_name, exp_pre_arr, exp_post_arr,
                sim_pre_arr, sim_post_arr,
                bins=args.bins, max_rate_display=args.max_rate_display,
            )
            _ratio_distribution_page(
                pdf, args.drug_name, exp_ratios_pu, sim_ratios_pu,
                bins=args.bins, max_log2_clip=args.max_log2_clip,
                exp_counts=exp_counts, sim_counts=sim_counts,
            )
        else:
            logger.info("Sim pkl paths not provided; skipping per-unit histogram pages.")
        _difference_vector_page(pdf, args.drug_name, feats, exp_vals, sim_vals,
                                max_log2_clip=args.max_log2_clip)
        _scatter_page(pdf, args.drug_name, feats, exp_vals, sim_vals,
                      max_log2_clip=args.max_log2_clip)
    logger.info("Wrote PDF: %s", out)

    sidecar = out.with_suffix(".json")
    payload = {
        "drug_name": args.drug_name,
        "features": list(feats),
        "experimental_ratios": {k: float(exp_ratios[k]) for k in feats},
        "simulated_ratios":    {k: float(sim_ratios[k]) for k in feats},
        "log2_difference":     {k: float(np.log2(max(sim_ratios[k], EPS))
                                          - np.log2(max(exp_ratios[k], EPS)))
                                for k in feats},
        "experimental_meta":   {k: (str(v) if isinstance(v, Path) else v)
                                for k, v in exp_meta.items()},
        "simulated_meta": {
            "baseline_fitness_json": str(args.baseline_sim_fitness),
            "drug_fitness_json":     str(args.drug_sim_fitness),
            "baseline_sim_pkl":      str(args.baseline_sim_pkl) if args.baseline_sim_pkl else None,
            "drug_sim_pkl":          str(args.drug_sim_pkl) if args.drug_sim_pkl else None,
        },
        "display_caps": {
            "max_ratio_display": args.max_ratio_display,
            "max_log2_clip":     args.max_log2_clip,
            "max_rate_display":  args.max_rate_display,
        },
    }
    if have_pkls:
        payload["per_unit_summary"] = {
            "experimental": {
                "n_common": exp_counts["total_common"],
                "n_both_active": exp_counts["both_active"],
                "n_silenced": exp_counts["silenced"],
                "n_activated": exp_counts["activated"],
                "median_ratio": float(np.median(exp_ratios_pu)) if exp_ratios_pu.size else None,
            },
            "simulated": {
                "n_common": sim_counts["total_common"],
                "n_both_active": sim_counts["both_active"],
                "n_silenced": sim_counts["silenced"],
                "n_activated": sim_counts["activated"],
                "median_ratio": float(np.median(sim_ratios_pu)) if sim_ratios_pu.size else None,
            },
        }
    with open(sidecar, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote sidecar JSON: %s", sidecar)


if __name__ == "__main__":
    main()
