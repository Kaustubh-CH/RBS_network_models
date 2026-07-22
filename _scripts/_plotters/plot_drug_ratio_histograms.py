"""Plot drug-response ratio histograms (baseline vs post-drug raw MEA) into one PDF.

Given a paired baseline + post-drug raw Maxwell recording (or pre-computed
threshold-crossings), this produces a single multi-page PDF visualising how the
network changed:

  Page 1  Summary — run metadata + the 12 population-level scalar ratios as a
          horizontal bar chart (post/pre), with a reference line at 1.0.
          These are exactly the ratios extract_drug_effects.py stamps into the
          target h5 and that the optimizer is fit against.
  Page 2  Per-channel firing-rate ratio histogram — the *distribution* of
          post/pre firing-rate change across all active electrodes, on a log2
          axis, split by the pre-drug E/I classification (top-20% firing rate
          -> inhibitory). This is the "histogram of ratios" in the literal
          per-electrode sense; the page-1 scalars are summaries of it.
  Page 3  Pre vs post per-channel firing-rate distributions (overlaid), so the
          shift underlying the ratios is visible directly.

All quantities reuse the helpers in extract_drug_effects.py so the figures match
what the extraction script writes to the target h5 (same window T, same E/I
split, same burst-detector params).

Usage (raw recordings):
  module load conda && conda activate preshifter
  HDF5_PLUGIN_PATH=/global/homes/k/ktub1999/hdf5_plugin_path_maxwell \
  python RBS_network_models/_scripts/plot_drug_ratio_histograms.py \
    --baseline_raw experimental_data/CDKL5_02.h5 \
    --drug_raw     experimental_data/Immediately_after_drug_11.h5 \
    --well well000 --drug_name bicuculline --n_jobs 1 \
    --baseline_target processed_experimental_targets/CDKL5_002_well000.h5 \
    --output drug_ratio_histograms_bicuculline.pdf

Usage (from existing spikesort_drug_comparison output):
  python RBS_network_models/_scripts/plot_drug_ratio_histograms.py \
    --from_existing_results MEA_Analysis/results_drug_comparison/CDKL5_02_vs_Imm11/well000/ \
    --drug_name bicuculline --output drug_ratio_histograms.pdf
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless / login node
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
# Reuse the extraction machinery so the figures match the stamped h5 exactly.
from extract_drug_effects import (  # noqa: E402
    EPS,
    INHIB_QUANTILE,
    RATIO_FEATURES,
    classify_channels_top20,
    compute_ratios,
    get_T_from_baseline_h5,
    load_from_existing_results,
    load_from_raw_recordings,
    _per_channel_firing_rates_hz,
    _truncate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("plot_drug_ratio_histograms")


def per_channel_stats(
    pre_units: Dict[str, np.ndarray],
    post_units: Dict[str, np.ndarray],
    T: float,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, dict]:
    """Per-channel firing-rate change over the channel intersection.

    Channels are partitioned by activity:
      both-active  pre>0 and post>0  -> a genuine post/pre ratio (no floor needed)
      silenced     pre>0 and post==0 -> reported as a count (would be ratio 0)
      activated    pre==0 and post>0 -> reported as a count (would be ratio inf)
      silent-both  pre==0 and post==0 -> excluded (no information)

    The ratio *histogram* uses only both-active channels so it isn't dominated by
    an overflow spike of silenced channels; the silenced/activated counts carry
    that information instead. Returns
    (pre_rates_all, post_rates_all, both_chans, both_ratios, counts), where the
    *_all arrays span every common channel (for the pre-vs-post distribution plot).
    """
    pre_rates = _per_channel_firing_rates_hz(pre_units, T)
    post_rates = _per_channel_firing_rates_hz(post_units, T)
    common = sorted(set(pre_rates) & set(post_rates))

    pre_all = np.array([pre_rates[c] for c in common])
    post_all = np.array([post_rates[c] for c in common])

    both_chans, ratios = [], []
    n_silenced = n_activated = n_silent_both = 0
    for ch in common:
        pr, po = pre_rates[ch], post_rates[ch]
        if pr > 0.0 and po > 0.0:
            both_chans.append(ch)
            ratios.append(po / pr)
        elif pr > 0.0:
            n_silenced += 1
        elif po > 0.0:
            n_activated += 1
        else:
            n_silent_both += 1
    counts = {
        "total_common": len(common),
        "both_active": len(both_chans),
        "silenced": n_silenced,
        "activated": n_activated,
        "silent_both": n_silent_both,
    }
    return pre_all, post_all, both_chans, np.asarray(ratios), counts


def _summary_page(
    pdf: PdfPages,
    drug_name: str,
    ratios: dict,
    diagnostics: dict,
    source_meta: dict,
    T: float,
) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(
        f"Drug-response ratios (post / pre) — {drug_name}",
        fontsize=15, fontweight="bold",
    )

    # Left: metadata text block.
    ax_txt = fig.add_axes([0.05, 0.08, 0.34, 0.80])
    ax_txt.axis("off")
    lines = [
        f"drug           : {drug_name}",
        f"source         : {source_meta.get('source', '?')}",
        f"window T (s)   : {T:.2f}",
        f"pre units      : {diagnostics.get('n_pre_units', '?')}",
        f"post units     : {diagnostics.get('n_post_units', '?')}",
        f"inhibitory ch  : {diagnostics.get('n_inhibitory_channels', '?')}",
        f"excitatory ch  : {diagnostics.get('n_excitatory_channels', '?')}",
        f"E/I threshold  : {diagnostics.get('ei_threshold_firing_rate_hz', float('nan')):.4f} Hz",
        f"  (top {(1.0 - INHIB_QUANTILE) * 100:.0f}% by pre firing rate -> inh)",
        f"pre spikes     : {diagnostics.get('pre_total_spikes', '?')}",
        f"post spikes    : {diagnostics.get('post_total_spikes', '?')}",
    ]
    for k in ("baseline_raw", "drug_raw", "well_dir", "well"):
        if k in source_meta:
            lines.append(f"{k:<14} : {os.path.basename(str(source_meta[k]))}")
    ax_txt.text(
        0.0, 1.0, "\n".join(lines), va="top", ha="left",
        family="monospace", fontsize=9.5, transform=ax_txt.transAxes,
    )

    # Right: horizontal bar chart of scalar ratios.
    ax = fig.add_axes([0.46, 0.08, 0.50, 0.80])
    feats = [f for f in RATIO_FEATURES if f in ratios]
    vals = [ratios[f] for f in feats]
    y = np.arange(len(feats))[::-1]  # first feature at top
    colors = ["#c0392b" if v > 1.0 else "#2471a3" for v in vals]
    ax.barh(y, vals, color=colors, height=0.7)
    ax.axvline(1.0, color="k", lw=1.2, ls="--", label="no change (1.0)")
    ax.set_yticks(y)
    ax.set_yticklabels([f.replace("_ratio", "") for f in feats], fontsize=9)
    ax.set_xlabel("post / pre")
    ax.set_title("Population-level ratio summary", fontsize=11)
    for yi, v in zip(y, vals):
        ax.text(v, yi, f" {v:.2f}", va="center", ha="left", fontsize=8)
    xmax = max(2.0, max(vals) * 1.15) if vals else 2.0
    ax.set_xlim(0, xmax)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    pdf.savefig(fig)
    plt.close(fig)


def _fmt_ratio_power(p: int) -> str:
    """Compact label for a log2 tick: -2 -> '1/4', 0 -> '1x', 4 -> '16x'."""
    if p == 0:
        return "1x"
    if p > 0:
        return f"{2 ** p}x"
    return f"1/{2 ** (-p)}"


def _ratio_histogram_page(
    pdf: PdfPages,
    drug_name: str,
    chans: List[str],
    ratios_pc: np.ndarray,
    classification: Dict[str, str],
    counts: dict,
    bins: int,
    max_span: float = 8.0,
) -> None:
    if ratios_pc.size == 0:
        return
    log2r_raw = np.log2(ratios_pc)
    is_inh = np.array([classification.get(ch) == "inhibitory" for ch in chans])

    # Display span from the central 1-99 percentile, capped so the long tail of
    # channels that go (near-)silent post-drug doesn't blow out the axis. Values
    # beyond the cap are clipped into the edge bins (the axis labels mark them as
    # "<=" the extremes via the tick text).
    lo, hi = np.percentile(log2r_raw, [1.0, 99.0])
    span = float(min(max_span, max(abs(lo), abs(hi), 2.0)))
    span = float(np.ceil(span))
    log2r = np.clip(log2r_raw, -span, span)

    fig, ax = plt.subplots(figsize=(11, 8.5))
    edges = np.linspace(-span, span, bins + 1)

    ax.hist(log2r, bins=edges, color="0.6", alpha=0.55, label=f"all ({log2r.size})")
    if is_inh.any():
        ax.hist(log2r[is_inh], bins=edges, histtype="step", lw=2.0,
                color="#8e44ad", label=f"inhibitory ({int(is_inh.sum())})")
    if (~is_inh).any():
        ax.hist(log2r[~is_inh], bins=edges, histtype="step", lw=2.0,
                color="#27ae60", label=f"excitatory ({int((~is_inh).sum())})")

    ax.axvline(0.0, color="k", lw=1.3, ls="--", label="no change (1x)")
    med = float(np.median(log2r_raw))  # median from unclipped data
    if -span <= med <= span:
        ax.axvline(med, color="#c0392b", lw=1.5, ls=":",
                   label=f"median = {2.0 ** med:.3g}x")

    # Label the log2 axis with compact multiplicative ratios; thin ticks so a
    # wide span stays legible (<= ~13 labels).
    step = max(1, int(np.ceil(span / 6.0)))
    tick_pow = np.arange(-int(span), int(span) + 1, step)
    ax.set_xticks(tick_pow)
    ax.set_xticklabels([_fmt_ratio_power(int(p)) for p in tick_pow])
    ax.set_xlim(-span, span)
    ax.set_xlabel(
        f"per-channel firing-rate ratio (post / pre, log2 axis; clipped to [1/{2 ** int(span)}, {2 ** int(span)}x])")
    ax.set_ylabel("number of channels")
    ax.set_title(
        f"Per-channel firing-rate ratio distribution — {drug_name}", fontsize=13)
    ax.set_ylabel("number of channels (active on both sides)")

    frac_up = float(np.mean(ratios_pc > 1.0)) if ratios_pc.size else 0.0
    ax.text(0.02, 0.97,
            f"both-active channels: {counts.get('both_active', ratios_pc.size)}\n"
            f"  {frac_up * 100:.0f}% increased / {(1 - frac_up) * 100:.0f}% decreased\n"
            f"silenced (pre>0, post=0): {counts.get('silenced', 0)}\n"
            f"activated (pre=0, post>0): {counts.get('activated', 0)}",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    pdf.savefig(fig)
    plt.close(fig)


def _pre_post_rate_page(
    pdf: PdfPages,
    drug_name: str,
    pre_rates: np.ndarray,
    post_rates: np.ndarray,
    bins: int,
) -> None:
    if pre_rates.size == 0 and post_rates.size == 0:
        return
    fig, ax = plt.subplots(figsize=(11, 8.5))
    pos = np.concatenate([pre_rates[pre_rates > 0], post_rates[post_rates > 0]])
    if pos.size == 0:
        plt.close(fig)
        return
    lo = max(pos.min(), 1e-3)
    hi = pos.max()
    edges = np.logspace(np.log10(lo), np.log10(hi), bins + 1)
    ax.hist(pre_rates[pre_rates > 0], bins=edges, alpha=0.5,
            color="#2471a3", label=f"pre  (median {np.median(pre_rates):.3f} Hz)")
    ax.hist(post_rates[post_rates > 0], bins=edges, alpha=0.5,
            color="#c0392b", label=f"post (median {np.median(post_rates):.3f} Hz)")
    ax.set_xscale("log")
    ax.set_xlabel("per-channel firing rate (Hz, log axis)")
    ax.set_ylabel("number of channels")
    ax.set_title(
        f"Per-channel firing-rate distributions, pre vs post — {drug_name}",
        fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline_raw", help="Baseline Maxwell .h5 raw recording.")
    parser.add_argument("--drug_raw", help="Post-drug Maxwell .h5 raw recording.")
    parser.add_argument("--well", default="well000", help="Maxwell stream id (default: well000).")
    parser.add_argument("--from_existing_results",
                        help="well_dir with pre_post_thresh_crossings.npz (mutually exclusive with --baseline_raw/--drug_raw).")
    parser.add_argument("--baseline_target",
                        help="Optional baseline target h5 to pull T_target_s from (for matching the optimizer window).")
    parser.add_argument("--drug_name", default="drug", help="Label used in titles (default: drug).")
    parser.add_argument("--threshold_mad", type=float, default=5.0,
                        help="Detection threshold (MAD units) for raw extraction (default: 5.0).")
    parser.add_argument("--peak_sign", choices=("neg", "pos", "both"), default="neg")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Workers for detect_peaks (default 1; >1 BrokenProcessPool on login nodes).")
    parser.add_argument("--T_seconds", type=float, default=None,
                        help="Comparison window. Default: T_target_s from --baseline_target, else min recording duration.")
    parser.add_argument("--bins", type=int, default=50, help="Histogram bins (default: 50).")
    parser.add_argument("--no_burst_ratios", action="store_true",
                        help="Skip burst detection; page-1 bar chart then shows only firing-rate ratios (fast).")
    parser.add_argument("-o", "--output", required=True, help="Output PDF path.")
    args = parser.parse_args()

    raw_provided = bool(args.baseline_raw or args.drug_raw)
    if raw_provided and args.from_existing_results:
        parser.error("--from_existing_results is mutually exclusive with --baseline_raw/--drug_raw.")
    if raw_provided and not (args.baseline_raw and args.drug_raw):
        parser.error("Both --baseline_raw and --drug_raw must be provided together.")

    if args.from_existing_results:
        pre_units, post_units, source_meta = load_from_existing_results(
            Path(args.from_existing_results).resolve())
    elif raw_provided:
        for p in (args.baseline_raw, args.drug_raw):
            if not Path(p).exists():
                parser.error(f"Raw recording not found: {p}")
        pre_units, post_units, source_meta = load_from_raw_recordings(
            baseline_raw=Path(args.baseline_raw).resolve(),
            drug_raw=Path(args.drug_raw).resolve(),
            well=args.well, threshold_mad=args.threshold_mad,
            peak_sign=args.peak_sign, n_jobs=args.n_jobs,
        )
    else:
        parser.error("Provide either --baseline_raw + --drug_raw, or --from_existing_results <well_dir>.")

    # Resolve comparison window T (same precedence as extract_drug_effects.py).
    T = args.T_seconds
    if T is None and args.baseline_target:
        T = get_T_from_baseline_h5(Path(args.baseline_target).resolve())
    if T is None:
        durations = []
        for u in (pre_units, post_units):
            if u:
                durations.append(max(t.max() for t in u.values() if len(t)))
        T = float(min(durations)) if durations else 0.0
        logger.warning("No T provided/found; defaulting to min recording duration: %.2f s", T)
    logger.info("Comparison window T = %.2f s", T)

    pre_clip = _truncate(pre_units, T)
    post_clip = _truncate(post_units, T)
    classification, ei_threshold = classify_channels_top20(pre_clip, T)

    pc_pre, pc_post, both_chans, pc_ratio, pc_counts = per_channel_stats(pre_clip, post_clip, T)
    logger.info(
        "Per-channel: %d common, %d both-active, %d silenced, %d activated, %d silent-both",
        pc_counts["total_common"], pc_counts["both_active"], pc_counts["silenced"],
        pc_counts["activated"], pc_counts["silent_both"],
    )

    # Scalar ratios + diagnostics. Burst detection is the slow part; allow skipping.
    if args.no_burst_ratios:
        from extract_drug_effects import _pop_firing_rate_hz, _class_mean_firing_rate_hz, _safe_ratio
        ratios = {
            "pop_FR_ratio": _safe_ratio(_pop_firing_rate_hz(post_clip, T), _pop_firing_rate_hz(pre_clip, T)),
            "exc_firing_rate_ratio": _safe_ratio(
                _class_mean_firing_rate_hz(post_clip, classification, "excitatory", T),
                _class_mean_firing_rate_hz(pre_clip, classification, "excitatory", T)),
            "inh_firing_rate_ratio": _safe_ratio(
                _class_mean_firing_rate_hz(post_clip, classification, "inhibitory", T),
                _class_mean_firing_rate_hz(pre_clip, classification, "inhibitory", T)),
        }
        n_inh = sum(1 for c in classification.values() if c == "inhibitory")
        diagnostics = {
            "n_pre_units": len(pre_clip), "n_post_units": len(post_clip),
            "n_inhibitory_channels": n_inh,
            "n_excitatory_channels": len(classification) - n_inh,
            "ei_threshold_firing_rate_hz": ei_threshold,
            "pre_total_spikes": int(sum(len(t) for t in pre_clip.values())),
            "post_total_spikes": int(sum(len(t) for t in post_clip.values())),
        }
    else:
        logger.info("Running burst detectors for scalar ratios (use --no_burst_ratios to skip)...")
        ratios, diagnostics = compute_ratios(pre_units, post_units, T)

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        _summary_page(pdf, args.drug_name, ratios, diagnostics, source_meta, T)
        _ratio_histogram_page(pdf, args.drug_name, both_chans, pc_ratio, classification, pc_counts, args.bins)
        _pre_post_rate_page(pdf, args.drug_name, pc_pre, pc_post, args.bins)
    logger.info("Wrote %d-page PDF: %s", 3, out)


if __name__ == "__main__":
    main()

'''
module load conda && conda activate preshifter
HDF5_PLUGIN_PATH=/global/homes/k/ktub1999/hdf5_plugin_path_maxwell \
python RBS_network_models/_scripts/plot_drug_ratio_histograms.py \
  --baseline_raw experimental_data/CDKL5_02.h5 \
  --drug_raw     experimental_data/Immediately_after_drug_11.h5 \
  --well well000 --drug_name bicuculline --n_jobs 1 \
  --baseline_target RBS_network_models/processed_experimental_targets/CDKL5_002_well000.h5 \
  --output drug_ratio_histograms_bicuculline.pdf

  '''