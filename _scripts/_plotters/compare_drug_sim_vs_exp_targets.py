"""Compare a simulated drug pre->post change against the experimental one, using
FIVE population features computed identically on every dataset.

The five features (defined ONCE in ``five_feature_metrics.compute_five_features``)
are:
    pop_fr_hz            mean population firing rate
    burst_freq_per_min   network bursts per minute
    synchrony_index      CV of the binned population spike-count signal
    ibi_s                mean inter-burst interval
    burst_duration_s     mean network-burst duration

The SAME function runs on all four spike-train datasets — experimental pre,
experimental post, simulated pre, simulated post — so "network burst",
"synchrony", etc. mean the same thing on both sides. (The earlier version read
precomputed scalars from the sim fitness json; this one recomputes from the sim
``trial_*_data.pkl`` spike trains, so the sim pkls are now REQUIRED.)

Inputs
------
Experimental — choose ONE source:
    (raw)    --baseline_raw / --drug_raw / --well   (channel-intersection MUA)
    (target) --baseline_target_h5 / --drug_target_h5 (spike-sorted unit_*/spike_times)

Simulated (REQUIRED):
    --baseline_sim_pkl / --drug_sim_pkl   trial_*_data.pkl (NetPyNE spkt/spkid)
    --baseline_sim_fitness / --drug_sim_fitness   optional, provenance only

Output: multi-page PDF + sidecar JSON with absolute pre/post values for all four
datasets, the post/pre ratios for exp and sim, and the log2 difference per feature.

Usage
-----
  module load conda && conda activate preshifter
  HDF5_PLUGIN_PATH=/global/homes/k/ktub1999/hdf5_plugin_path_maxwell \\
  python RBS_network_models/_scripts/compare_drug_sim_vs_exp_targets.py \\
    --baseline_raw experimental_data/CDKL5_02.h5 \\
    --drug_raw     experimental_data/Immediately_after_drug_11.h5 \\
    --well well001 --n_jobs 1 \\
    --baseline_sim_pkl <.../gen_7/trial_7_data.pkl> \\
    --drug_sim_pkl     <.../gen_6/trial_6_data.pkl> \\
    --drug_name bicuculline \\
    --output compare_5feat_bicuculline.pdf
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
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from extract_drug_effects import (  # noqa: E402
    EPS,
    load_from_raw_recordings,
    get_T_from_baseline_h5,
    _truncate,
)
from five_feature_metrics import (  # noqa: E402
    FEATURE_KEYS,
    FEATURE_LABELS,
    SYNC_BIN_MS,
    BURST_KW,
    compute_five_features,
    ratios_from_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compare_drug_sim_vs_exp_targets")


# ---------- spike-train loaders -------------------------------------------

def load_units_from_target_h5(path: Path) -> Tuple[Dict[str, np.ndarray], float]:
    """Read every ``unit_<id>/spike_times`` (seconds) from a processed target h5.

    Returns ({unit_id: spike_time_array}, max_spike_time). Empty units are kept
    so the population size matches the recording.
    """
    units: Dict[str, np.ndarray] = {}
    tmax = 0.0
    with h5py.File(path, "r") as f:
        for key in f.keys():
            if not key.startswith("unit_"):
                continue
            grp = f[key]
            if "spike_times" not in grp:
                continue
            st = np.asarray(grp["spike_times"][:], dtype=float)
            units[key] = st
            if st.size:
                tmax = max(tmax, float(st.max()))
    return units, tmax


def load_sim_spike_trains(pkl_path: Path) -> Tuple[Dict[str, np.ndarray], float, dict]:
    """NetPyNE trial_*_data.pkl -> ({gid: spike_times_seconds}, T_seconds, meta).

    ALL cells are included (silent cells get empty arrays) so pop_FR and burst
    participation divide by the true population size, matching how the
    experimental side keeps silent channels/units. spkt is in ms; T from
    simConfig['duration'] (ms).
    """
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    sd = d.get("simData", d.get("allSimData", d))
    spkt = np.asarray(sd.get("spkt", sd.get("spkts", [])), dtype=float) / 1000.0  # ms -> s
    spkid = np.asarray(sd.get("spkid", sd.get("spkids", [])), dtype=int)

    # Total cell gids: prefer the explicit cell list; fall back to max active id.
    gids = None
    net = d.get("net")
    if isinstance(net, dict) and isinstance(net.get("cells"), list) and net["cells"]:
        try:
            gids = [int(c["gid"]) for c in net["cells"] if isinstance(c, dict) and "gid" in c]
        except Exception:
            gids = None
    if not gids:
        max_gid = int(spkid.max()) if spkid.size else -1
        gids = list(range(max_gid + 1))

    duration_ms = None
    if isinstance(d.get("simConfig"), dict):
        duration_ms = d["simConfig"].get("duration")
    if duration_ms is None:
        duration_ms = sd.get("simDuration") or sd.get("duration")
    if duration_ms is None:
        duration_ms = float(spkt.max() * 1000.0) if spkt.size else 1000.0
    T = float(duration_ms) / 1000.0

    units: Dict[str, np.ndarray] = {str(g): np.empty(0, dtype=float) for g in gids}
    if spkt.size and spkid.size:
        order = np.argsort(spkid, kind="stable")
        sid = spkid[order]
        stt = spkt[order]
        uniq, starts = np.unique(sid, return_index=True)
        starts = list(starts) + [len(sid)]
        for i, g in enumerate(uniq):
            units[str(int(g))] = stt[starts[i]:starts[i + 1]]
    n_active = sum(1 for v in units.values() if v.size)
    meta = {"n_cells": len(units), "n_active": n_active, "_source": str(pkl_path)}
    return units, T, meta


# ---------- plotting -------------------------------------------------------

_EXP_PRE_C, _EXP_POST_C = "#85c1e9", "#1b4f72"   # light/dark blue
_SIM_PRE_C, _SIM_POST_C = "#f1948a", "#922b21"   # light/dark red


def _summary_text_page(pdf, drug_name, exp_pre, exp_post, sim_pre, sim_post,
                       exp_ratios, sim_ratios, exp_meta, sim_meta):
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(f"Pre -> post comparison ({drug_name}) — five features\n"
                 "experimental vs simulated",
                 fontsize=14, fontweight="bold")
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.84])
    ax.axis("off")

    def _g(d, k):
        return d.get(k, float("nan"))

    lines = []
    lines.append("DATASETS")
    lines.append(f"  experimental source : {exp_meta.get('source','?')}")
    lines.append(f"    pre  : {os.path.basename(str(exp_meta.get('baseline','')))}  "
                 f"T={exp_pre['_T_seconds']:.2f}s  units={exp_pre['_n_units']}  "
                 f"spikes={exp_pre['_total_spikes']}  net-bursts={exp_pre['_n_network_bursts']}")
    lines.append(f"    post : {os.path.basename(str(exp_meta.get('drug','')))}  "
                 f"T={exp_post['_T_seconds']:.2f}s  units={exp_post['_n_units']}  "
                 f"spikes={exp_post['_total_spikes']}  net-bursts={exp_post['_n_network_bursts']}")
    lines.append(f"  simulated")
    lines.append(f"    pre  : {os.path.basename(str(sim_meta.get('baseline','')))}  "
                 f"T={sim_pre['_T_seconds']:.2f}s  cells={sim_pre['_n_units']}  "
                 f"spikes={sim_pre['_total_spikes']}  net-bursts={sim_pre['_n_network_bursts']}")
    lines.append(f"    post : {os.path.basename(str(sim_meta.get('drug','')))}  "
                 f"T={sim_post['_T_seconds']:.2f}s  cells={sim_post['_n_units']}  "
                 f"spikes={sim_post['_total_spikes']}  net-bursts={sim_post['_n_network_bursts']}")
    lines.append("")
    lines.append("DEFINITIONS (computed identically on all four datasets)")
    lines.append(f"  network burst   : compute_network_bursts(**{BURST_KW})")
    lines.append(f"  synchrony index : CV (std/mean) of population spike count, "
                 f"{SYNC_BIN_MS:g} ms bins")
    lines.append(f"  burst freq      : network bursts / T * 60  (per minute)")
    lines.append("")
    header = (f"  {'feature':<28} {'exp_pre':>10} {'exp_post':>10} {'exp r':>8}   "
              f"{'sim_pre':>10} {'sim_post':>10} {'sim r':>8}")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for k in FEATURE_KEYS:
        lines.append(
            f"  {FEATURE_LABELS[k]:<28} {_g(exp_pre,k):>10.4f} {_g(exp_post,k):>10.4f} "
            f"{exp_ratios[k]:>8.3f}   {_g(sim_pre,k):>10.4f} {_g(sim_post,k):>10.4f} "
            f"{sim_ratios[k]:>8.3f}")
    lines.append("")
    lines.append("  (r = post/pre ratio; agreement = exp r and sim r move the same way)")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
            family="monospace", fontsize=9, transform=ax.transAxes)
    pdf.savefig(fig)
    plt.close(fig)


def _absolute_bars_page(pdf, drug_name, exp_pre, exp_post, sim_pre, sim_post):
    """One panel per feature; 4 bars (exp pre/post, sim pre/post) on its own scale."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    fig.suptitle(f"Absolute pre vs post per feature ({drug_name})",
                 fontsize=14, fontweight="bold")
    axes = axes.ravel()
    labels = ["exp pre", "exp post", "sim pre", "sim post"]
    colors = [_EXP_PRE_C, _EXP_POST_C, _SIM_PRE_C, _SIM_POST_C]
    for ax, k in zip(axes, FEATURE_KEYS):
        vals = [exp_pre[k], exp_post[k], sim_pre[k], sim_post[k]]
        x = np.arange(4)
        ax.bar(x, vals, color=colors, width=0.7)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=20)
        ax.set_title(FEATURE_LABELS[k], fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        top = max(vals) if max(vals) > 0 else 1.0
        ax.set_ylim(0, top * 1.18)
    # last (6th) panel: legend / note
    axes[-1].axis("off")
    axes[-1].text(0.5, 0.5,
                  "blue = experimental\nred = simulated\nlight = pre,  dark = post",
                  ha="center", va="center", fontsize=11,
                  transform=axes[-1].transAxes)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


# Short labels for the compact unity-scatter annotations.
_SHORT_LABEL = {
    "pop_fr_hz": "pop FR",
    "burst_freq_per_min": "burst freq",
    "synchrony_index": "synchrony",
    "ibi_s": "IBI",
    "burst_duration_s": "burst dur",
}
# A metric "agrees" if sim_r/exp_r is within ±25% (both ratios point the same
# way to within a quarter). Symmetric band on a multiplicative axis.
_AGREE_LO, _AGREE_HI = 1.0 / 1.25, 1.25


def _agreement_stats(exp_vals, sim_vals):
    """Return (n_within_25pct, n_total, pearson_r_log2, mean_abs_log2_dev)."""
    rel = sim_vals / np.maximum(exp_vals, EPS)
    within = int(np.sum((rel >= _AGREE_LO) & (rel <= _AGREE_HI)))
    lx = np.log2(np.maximum(exp_vals, EPS))
    ly = np.log2(np.maximum(sim_vals, EPS))
    if len(lx) > 1 and np.std(lx) > 0 and np.std(ly) > 0:
        r = float(np.corrcoef(lx, ly)[0, 1])
    else:
        r = float("nan")
    mad_log2 = float(np.mean(np.abs(ly - lx))) if len(lx) else float("nan")
    return within, len(exp_vals), r, mad_log2


def _ratio_compare_page(pdf, drug_name, exp_ratios, sim_ratios, max_ratio_display):
    """Unity-line agreement scatter (exp vs sim post/pre) + log2 difference."""
    feats = list(FEATURE_KEYS)
    exp_vals = np.array([exp_ratios[k] for k in feats], dtype=float)
    sim_vals = np.array([sim_ratios[k] for k in feats], dtype=float)
    within, n_tot, pearson_r, mad_log2 = _agreement_stats(exp_vals, sim_vals)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7.5))
    fig.suptitle(f"Post/pre ratio: experimental vs simulated ({drug_name})",
                 fontsize=14, fontweight="bold")

    # --- left: unity-line agreement scatter (log2 axes) ---
    allv = np.concatenate([exp_vals, sim_vals, [0.5, 2.0]])
    lo = float(np.min(allv)) / 1.3
    hi = float(np.max(allv)) * 1.3
    edge = np.array([lo, hi])
    ax1.plot(edge, edge, "k--", lw=1.3, zorder=1, label="unity (sim = exp)")
    ax1.axhline(1.0, color="0.6", lw=0.8, ls=":", zorder=1)
    ax1.axvline(1.0, color="0.6", lw=0.8, ls=":", zorder=1)
    ax1.scatter(exp_vals, sim_vals, s=90, c="#2c3e50", zorder=3,
                edgecolor="k", linewidth=0.5)
    for x, yv, k in zip(exp_vals, sim_vals, feats):
        ax1.annotate(_SHORT_LABEL[k], (x, yv), xytext=(6, 4),
                     textcoords="offset points", fontsize=9)
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log", base=2)
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("experimental post/pre ratio")
    ax1.set_ylabel("simulated post/pre ratio")
    ax1.set_title("agreement (points on the line = match)", fontsize=11)
    ax1.legend(loc="lower right", fontsize=8)
    ax1.text(0.03, 0.97,
             f"Pearson r (log2 ratios): {pearson_r:.3f}",
             transform=ax1.transAxes, va="top", ha="left", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    y = np.arange(len(feats))[::-1]
    diff = np.log2(np.maximum(sim_vals, EPS)) - np.log2(np.maximum(exp_vals, EPS))
    colors = ["#c0392b" if v > 0 else "#2471a3" for v in diff]
    ax2.barh(y, diff, color=colors, height=0.6)
    ax2.axvline(0.0, color="k", lw=1.2, ls="--")
    ax2.set_yticks(y)
    ax2.set_yticklabels([FEATURE_LABELS[k] for k in feats], fontsize=9)
    ax2.set_xlabel("log2(sim ratio) - log2(exp ratio)\n(>0 sim over, <0 sim under)")
    ax2.set_title("change-vector difference (sim - exp)", fontsize=11)
    for yi, vr in zip(y, diff):
        ax2.text(vr, yi, f" {vr:+.2f} ({2.0**vr:.2f}x)", va="center",
                 ha="left" if vr >= 0 else "right", fontsize=8)
    span = max(1.0, float(np.abs(diff).max()) * 1.3) if diff.size else 1.0
    ax2.set_xlim(-span, span)
    rms = float(np.sqrt(np.mean(diff ** 2))) if diff.size else float("nan")
    n_agree = int(np.sum(np.sign(sim_vals - 1.0) == np.sign(exp_vals - 1.0)))
    ax2.text(0.02, 0.02,
             f"RMS log2 deviation: {rms:.3f}  ({2.0**rms:.2f}x)\n"
             f"same-direction features: {n_agree}/{len(feats)}",
             transform=ax2.transAxes, va="bottom", ha="left", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax2.grid(axis="x", alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close(fig)


# ---------- main -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Experimental side — choose ONE source.
    parser.add_argument("--baseline_target_h5", help="Pre-drug target h5 (unit_*/spike_times).")
    parser.add_argument("--drug_target_h5", help="Post-drug target h5 (unit_*/spike_times).")
    parser.add_argument("--baseline_raw", help="Pre-drug raw Maxwell .h5.")
    parser.add_argument("--drug_raw", help="Post-drug raw Maxwell .h5.")
    parser.add_argument("--well", default="well000")
    parser.add_argument("--threshold_mad", type=float, default=5.0)
    parser.add_argument("--peak_sign", choices=("neg", "pos", "both"), default="neg")
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--baseline_target_for_T",
                        help="Optional baseline target h5 to read T_target_s from (raw mode).")
    parser.add_argument("--T_seconds", type=float, default=None,
                        help="Experimental window; default = min(max spike time) both sides.")
    # Simulated side (pkls required — we recompute features from spike trains).
    parser.add_argument("--baseline_sim_pkl", required=True,
                        help="trial_*_data.pkl baseline sim (NetPyNE spkt/spkid).")
    parser.add_argument("--drug_sim_pkl", required=True,
                        help="trial_*_data.pkl drug sim.")
    parser.add_argument("--baseline_sim_fitness", help="Optional, provenance only.")
    parser.add_argument("--drug_sim_fitness", help="Optional, provenance only.")
    parser.add_argument("--max_ratio_display", type=float, default=8.0)
    parser.add_argument("--drug_name", default="drug")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    raw_mode = bool(args.baseline_raw or args.drug_raw)
    tgt_mode = bool(args.baseline_target_h5 or args.drug_target_h5)
    if raw_mode == tgt_mode:
        parser.error("Provide exactly ONE experimental source: "
                     "--baseline_raw/--drug_raw OR --baseline_target_h5/--drug_target_h5.")
    if raw_mode and not (args.baseline_raw and args.drug_raw):
        parser.error("Raw mode needs BOTH --baseline_raw and --drug_raw.")
    if tgt_mode and not (args.baseline_target_h5 and args.drug_target_h5):
        parser.error("Target mode needs BOTH --baseline_target_h5 and --drug_target_h5.")

    exp_paths = ([args.baseline_raw, args.drug_raw] if raw_mode
                 else [args.baseline_target_h5, args.drug_target_h5])
    for p in exp_paths + [args.baseline_sim_pkl, args.drug_sim_pkl]:
        if not Path(p).exists():
            parser.error(f"File not found: {p}")

    # --- experimental spike trains ---
    if raw_mode:
        logger.info("Experimental source: RAW (channel intersection, %.1f MAD, "
                    "peak_sign=%s, n_jobs=%d, well=%s)",
                    args.threshold_mad, args.peak_sign, args.n_jobs, args.well)
        pre_units, post_units, _src = load_from_raw_recordings(
            baseline_raw=Path(args.baseline_raw).resolve(),
            drug_raw=Path(args.drug_raw).resolve(),
            well=args.well, threshold_mad=args.threshold_mad,
            peak_sign=args.peak_sign, n_jobs=args.n_jobs,
        )
        exp_baseline_path, exp_drug_path = args.baseline_raw, args.drug_raw
        exp_source_label = "raw_recordings (channel intersection)"
    else:
        pre_units, _ = load_units_from_target_h5(Path(args.baseline_target_h5).resolve())
        post_units, _ = load_units_from_target_h5(Path(args.drug_target_h5).resolve())
        exp_baseline_path, exp_drug_path = args.baseline_target_h5, args.drug_target_h5
        exp_source_label = "processed_target_h5 (spike-sorted units)"

    pre_tmax = max((t.max() for t in pre_units.values() if len(t)), default=0.0)
    post_tmax = max((t.max() for t in post_units.values() if len(t)), default=0.0)
    if args.T_seconds is not None:
        T_exp = args.T_seconds
    elif args.baseline_target_for_T:
        T_exp = float(get_T_from_baseline_h5(Path(args.baseline_target_for_T).resolve()))
    else:
        T_exp = float(min(pre_tmax, post_tmax))
    if T_exp <= 0.0:
        parser.error("Could not determine a positive experimental T window.")
    logger.info("Experimental window T = %.2f s (pre tmax=%.2f, post tmax=%.2f)",
                T_exp, pre_tmax, post_tmax)

    exp_pre = compute_five_features(_truncate(pre_units, T_exp), T_exp, "exp:pre")
    exp_post = compute_five_features(_truncate(post_units, T_exp), T_exp, "exp:post")

    # --- simulated spike trains ---
    sim_pre_units, T_sim_pre, sim_pre_meta = load_sim_spike_trains(
        Path(args.baseline_sim_pkl).resolve())
    sim_post_units, T_sim_post, sim_post_meta = load_sim_spike_trains(
        Path(args.drug_sim_pkl).resolve())
    logger.info("Simulated windows: pre T=%.2fs (%d cells), post T=%.2fs (%d cells)",
                T_sim_pre, sim_pre_meta["n_cells"], T_sim_post, sim_post_meta["n_cells"])
    sim_pre = compute_five_features(_truncate(sim_pre_units, T_sim_pre), T_sim_pre, "sim:pre")
    sim_post = compute_five_features(_truncate(sim_post_units, T_sim_post), T_sim_post, "sim:post")

    exp_ratios = ratios_from_features(exp_pre, exp_post)
    sim_ratios = ratios_from_features(sim_pre, sim_post)
    logger.info("Experimental ratios: " +
                "  ".join(f"{k}={exp_ratios[k]:.3f}" for k in FEATURE_KEYS))
    logger.info("Simulated   ratios: " +
                "  ".join(f"{k}={sim_ratios[k]:.3f}" for k in FEATURE_KEYS))
    _exp_r_arr = np.array([exp_ratios[k] for k in FEATURE_KEYS], dtype=float)
    _sim_r_arr = np.array([sim_ratios[k] for k in FEATURE_KEYS], dtype=float)
    _within, _ntot, _pearson, _mad = _agreement_stats(_exp_r_arr, _sim_r_arr)
    logger.info("Agreement: %d/%d within ±25%% of unity, Pearson r(log2)=%.3f, "
                "mean|log2| dev=%.3f", _within, _ntot, _pearson, _mad)

    exp_meta = {"source": exp_source_label,
                "baseline": exp_baseline_path, "drug": exp_drug_path}
    sim_meta = {"baseline": args.baseline_sim_pkl, "drug": args.drug_sim_pkl}

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        _summary_text_page(pdf, args.drug_name, exp_pre, exp_post, sim_pre, sim_post,
                           exp_ratios, sim_ratios, exp_meta, sim_meta)
        _absolute_bars_page(pdf, args.drug_name, exp_pre, exp_post, sim_pre, sim_post)
        _ratio_compare_page(pdf, args.drug_name, exp_ratios, sim_ratios,
                            args.max_ratio_display)
    logger.info("Wrote PDF: %s", out)

    def _clean(d):
        return {k: float(v) for k, v in d.items()}

    sidecar = out.with_suffix(".json")
    payload = {
        "drug_name": args.drug_name,
        "features": list(FEATURE_KEYS),
        "feature_labels": FEATURE_LABELS,
        "definitions": {
            "burst_detector_kwargs": BURST_KW,
            "synchrony_bin_ms": SYNC_BIN_MS,
            "burst_freq": "n_network_bursts / T * 60 (per minute)",
            "note": "all five features computed by five_feature_metrics.compute_five_features "
                    "identically on exp pre/post and sim pre/post",
        },
        "experimental": {
            "source": exp_source_label,
            "baseline": str(exp_baseline_path),
            "drug": str(exp_drug_path),
            "pre": _clean(exp_pre), "post": _clean(exp_post),
            "ratios": {k: float(exp_ratios[k]) for k in FEATURE_KEYS},
        },
        "simulated": {
            "baseline_pkl": str(args.baseline_sim_pkl),
            "drug_pkl": str(args.drug_sim_pkl),
            "baseline_fitness": str(args.baseline_sim_fitness) if args.baseline_sim_fitness else None,
            "drug_fitness": str(args.drug_sim_fitness) if args.drug_sim_fitness else None,
            "pre": _clean(sim_pre), "post": _clean(sim_post),
            "ratios": {k: float(sim_ratios[k]) for k in FEATURE_KEYS},
        },
        "log2_difference": {
            k: float(np.log2(max(sim_ratios[k], EPS)) - np.log2(max(exp_ratios[k], EPS)))
            for k in FEATURE_KEYS
        },
        "agreement": {
            "n_within_25pct_of_unity": _within,
            "n_total": _ntot,
            "band": [_AGREE_LO, _AGREE_HI],
            "pearson_r_log2": (None if np.isnan(_pearson) else _pearson),
            "mean_abs_log2_deviation": _mad,
        },
    }
    with open(sidecar, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote sidecar JSON: %s", sidecar)


if __name__ == "__main__":
    main()
