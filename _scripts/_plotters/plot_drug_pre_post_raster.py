#!/usr/bin/env python3
"""Plot pre vs post (drug) raster + network-activity for a 20 s window, reusing
the same spike detection and E/I classification as extract_drug_effects.py.

Detects threshold-crossings on the pre/post channel intersection (via
extract_drug_effects.load_from_raw_recordings), classifies E/I from the pre side
(top-20% firing rate -> inhibitory, same as the ratio extraction), truncates to
--T_seconds, and renders a 2x2 figure:

    row 0 = PRE (before drug)   row 1 = POST (after drug)
    col 0 = raster (E blue / I red, channels grouped by class then rate)
    col 1 = population firing rate over time (total + E + I), shared y-limits

Usage (run inside an allocation; set HDF5_PLUGIN_PATH before python):
    python plot_drug_pre_post_raster.py \
        --baseline_raw experimental_data/CDKL5_02.h5 \
        --drug_raw experimental_data/Immediately_after_drug_11.h5 \
        --well well001 --T_seconds 20 --drug_name bicuculline \
        --output .../CDKL5_002_well001_bicuculline_20s_pre_post.png
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from extract_drug_effects import (  # noqa: E402
    load_from_raw_recordings,
    classify_channels_top20,
    _truncate,
    _pop_firing_rate_hz,
)


def binned_rate_hz(units, T, bin_s, channels=None):
    """Mean per-unit firing rate (Hz) in time bins over the given channel subset."""
    edges = np.arange(0.0, T + bin_s, bin_s)
    centers = 0.5 * (edges[:-1] + edges[1:])
    chans = list(units.keys()) if channels is None else [c for c in channels if c in units]
    n = max(len(chans), 1)
    counts = np.zeros(len(edges) - 1)
    for c in chans:
        t = units[c]
        t = t[t <= T]
        if len(t):
            counts += np.histogram(t, bins=edges)[0]
    return centers, counts / (n * bin_s)


def raster_order(units, classification, T):
    """Return channel ids ordered: inhibitory block then excitatory block, each
    sorted by firing rate ascending, so structure is visible on the y-axis."""
    def rate(c):
        return len(units[c][units[c] <= T])
    inh = sorted([c for c in units if classification.get(c) == "inhibitory"], key=rate)
    exc = sorted([c for c in units if classification.get(c) == "excitatory"], key=rate)
    return inh + exc, len(inh)


def plot_raster(ax, units, order, n_inh, T, title):
    idx = {c: i for i, c in enumerate(order)}
    # excitatory (blue) then inhibitory (red) so both are visible
    for c in order:
        t = units[c][units[c] <= T]
        if not len(t):
            continue
        y = np.full(len(t), idx[c])
        color = "#c0392b" if idx[c] < n_inh else "#2c5fb8"
        ax.scatter(t, y, s=0.6, c=color, marker="|", linewidths=0.3, rasterized=True)
    ax.axhline(n_inh - 0.5, color="k", lw=0.6, ls="--", alpha=0.5)
    ax.set_xlim(0, T)
    ax.set_ylim(-1, len(order))
    ax.set_ylabel("channel (I below dash, E above)")
    ax.set_title(title, fontsize=10)


def build_figure(pre_units, post_units, T, bin_s, bin_ms, drug_name, well, output):
    """Render the 2x2 pre/post raster + network-rate figure for one window T."""
    pre = _truncate(pre_units, T)
    post = _truncate(post_units, T)

    # E/I classification from the PRE side (identical rule to the ratio extraction)
    classification, ei_thr = classify_channels_top20(pre, T)
    inh_ch = [c for c, v in classification.items() if v == "inhibitory"]
    exc_ch = [c for c, v in classification.items() if v == "excitatory"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9),
                             gridspec_kw={"width_ratios": [1.6, 1.0]})

    for row, (units, label) in enumerate([(pre, "PRE (before)"), (post, "POST (after)")]):
        order, n_inh = raster_order(units, classification, T)
        plot_raster(axes[row, 0], units, order, n_inh, T,
                    "%s — %s well %s (T=%gs)" % (label, drug_name, well, T))

        ax = axes[row, 1]
        c_all, r_all = binned_rate_hz(units, T, bin_s)
        c_e, r_e = binned_rate_hz(units, T, bin_s, channels=exc_ch)
        c_i, r_i = binned_rate_hz(units, T, bin_s, channels=inh_ch)
        ax.plot(c_all, r_all, color="k", lw=1.0, label="all")
        ax.plot(c_e, r_e, color="#2c5fb8", lw=0.9, label="exc")
        ax.plot(c_i, r_i, color="#c0392b", lw=0.9, label="inh")
        ax.set_xlim(0, T)
        ax.set_ylabel("mean FR (Hz, %gms bins)" % bin_ms)
        ax.set_title("%s network activity  (popFR=%.2f Hz)"
                     % (label, _pop_firing_rate_hz(units, T)), fontsize=10)
        ax.legend(fontsize=8, loc="upper right")

    # shared network-activity y-limits for fair pre/post comparison
    ymax = max(axes[0, 1].get_ylim()[1], axes[1, 1].get_ylim()[1])
    axes[0, 1].set_ylim(0, ymax)
    axes[1, 1].set_ylim(0, ymax)
    for ax in axes[1, :]:
        ax.set_xlabel("time (s)")

    fig.suptitle(
        "%s pre/post drug — well %s — %d units (%d I / %d E), E/I thr=%.2f Hz"
        % (drug_name, well, len(pre), len(inh_ch), len(exc_ch), ei_thr),
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output, dpi=130)
    plt.close(fig)
    print("Saved figure -> %s" % output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_raw", required=True)
    p.add_argument("--drug_raw", required=True)
    p.add_argument("--well", default="well001")
    p.add_argument("--T_seconds", default="20",
                   help="Window length(s) in seconds; comma-separated for multiple "
                        "figures from one detection pass (e.g. '20,60').")
    p.add_argument("--bin_ms", type=float, default=50.0)
    p.add_argument("--threshold_mad", type=float, default=5.0)
    p.add_argument("--peak_sign", default="neg")
    p.add_argument("--n_jobs", type=int, default=1)
    p.add_argument("--drug_name", default="bicuculline")
    p.add_argument("--output_dir", required=True,
                   help="Directory for output figures.")
    p.add_argument("--output_prefix", required=True,
                   help="Filename prefix; each figure is <prefix>_<T>s_pre_post.png")
    args = p.parse_args()

    T_list = [float(t.strip()) for t in args.T_seconds.split(",") if t.strip()]
    bin_s = args.bin_ms / 1000.0

    # Detect once on the full recordings; each T just truncates + plots.
    pre_units, post_units, meta = load_from_raw_recordings(
        baseline_raw=Path(args.baseline_raw).resolve(),
        drug_raw=Path(args.drug_raw).resolve(),
        well=args.well,
        threshold_mad=args.threshold_mad,
        peak_sign=args.peak_sign,
        n_jobs=args.n_jobs,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    for T in T_list:
        out = os.path.join(args.output_dir,
                           "%s_%gs_pre_post.png" % (args.output_prefix, T))
        build_figure(pre_units, post_units, T, bin_s, args.bin_ms,
                     args.drug_name, args.well, out)


if __name__ == "__main__":
    main()
