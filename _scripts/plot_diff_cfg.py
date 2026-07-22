#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def flatten_dict(d, parent_key="", sep="."):
    """Flatten nested dictionaries into dot-separated keys."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def load_config(path, section=None):
    with open(path, "r") as f:
        data = json.load(f)
    if section:
        # section can be nested like "simConfig" or "a.b.c"
        for part in section.split("."):
            data = data[part]
    return data


def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def compare_numeric_params(cfg1, cfg2):
    f1 = flatten_dict(cfg1)
    f2 = flatten_dict(cfg2)

    common_keys = sorted(set(f1.keys()) & set(f2.keys()))
    rows = []
    for k in common_keys:
        v1, v2 = f1[k], f2[k]
        if is_number(v1) and is_number(v2):
            if np.isfinite(v1) and np.isfinite(v2):
                rows.append((k, float(v1), float(v2)))
    return rows


def plot_comparison(rows, label1, label2, top_n=30, out_png="config_compare.png"):
    if not rows:
        raise ValueError("No common numeric parameters found.")

    # Sort by absolute difference and keep top_n
    rows = sorted(rows, key=lambda x: abs(x[2] - x[1]), reverse=True)[:top_n]
    keys = [r[0] for r in rows]
    v1 = np.array([r[1] for r in rows], dtype=float)
    v2 = np.array([r[2] for r in rows], dtype=float)

    # Percent change: (v2 - v1) / |v1|
    denom = np.where(np.abs(v1) < 1e-12, np.nan, np.abs(v1))
    pct = (v2 - v1) / denom * 100.0

    fig = plt.figure(figsize=(16, max(7, 0.35 * len(keys))))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.3], wspace=0.25)

    # Left: side-by-side bars
    ax1 = fig.add_subplot(gs[0, 0])
    y = np.arange(len(keys))
    h = 0.38
    ax1.barh(y - h / 2, v1, h, label=label1, alpha=0.85)
    ax1.barh(y + h / 2, v2, h, label=label2, alpha=0.85)
    ax1.set_yticks(y)
    ax1.set_yticklabels(keys, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("Parameter value")
    ax1.set_title("Network parameter values (top differences)")
    ax1.legend()
    ax1.grid(axis="x", linestyle="--", alpha=0.3)

    # Right: percent change
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    colors = np.where(np.nan_to_num(pct, nan=0.0) >= 0, "tab:green", "tab:red")
    ax2.barh(y, np.nan_to_num(pct, nan=0.0), color=colors, alpha=0.85)
    ax2.axvline(0, color="black", linewidth=1)
    ax2.set_xlabel("% change")
    ax2.set_title(f"% change ({label2} vs {label1})")
    ax2.grid(axis="x", linestyle="--", alpha=0.3)
    plt.setp(ax2.get_yticklabels(), visible=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {out_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two configuration JSON files and plot differing numeric parameters."
    )
    parser.add_argument("json1", help="Path to first config JSON")
    parser.add_argument("json2", help="Path to second config JSON")
    parser.add_argument(
        "--section",
        default="simConfig",
        help="Optional section to compare (default: simConfig). Example: simConfig",
    )
    parser.add_argument("--label1", default="Config A", help="Legend label for first config")
    parser.add_argument("--label2", default="Config B", help="Legend label for second config")
    parser.add_argument("--top-n", type=int, default=30, help="Number of top differing params")
    parser.add_argument("--out", default="config_compare.png", help="Output PNG file")
    args = parser.parse_args()

    cfg1 = load_config(args.json1, section=args.section if args.section else None)
    cfg2 = load_config(args.json2, section=args.section if args.section else None)

    rows = compare_numeric_params(cfg1, cfg2)
    plot_comparison(rows, args.label1, args.label2, top_n=args.top_n, out_png=args.out)


if __name__ == "__main__":
    main()