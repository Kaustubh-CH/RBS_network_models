#!/usr/bin/env python3
"""
feature_window.py

Helper module + diagnostic CLI for rate-feature window analysis.

ROLE 1 — importable helpers (used by create_experimental_target.py):
  - truncate(spike_times, T)
  - firing_rates(truncated, cell_types, T)
  - burst_rates(truncated)                          (production-matching detector params)
  - compute_curve(spike_times, cell_types, T_grid)  → rate features per T
  - build_grid(user_grid, T_full)                   (always appends T_full)
  - load_schema(path)                               (importlib-based fit_schema loader)
  - schema_fitness_curve(fit_schema, rows)          (rate-error fitness sum vs T_full)
  - SCHEMA_RATE_MAP, MAIN_BURST_KW, PRE_BURST_KW, FEATURES, DEFAULT_T_GRID,
    DEFAULT_SCHEMAS, DEFAULT_FOCUS_SCHEMA

ROLE 2 — CLI: produces a 3-page PDF for a single well's mea_analysis_routine
outputs (spike_times.npy + metrics_curated.xlsx).

Rate features (counts excluded by design):
  - exc_firing_rate_hz, inh_firing_rate_hz, mean_firing_rate_hz
  - burstlet_rate_hz       (compute_network_bursts main detection: thr=40, mp=0.05)
  - network_burst_rate_hz, superburst_rate_hz       (same main detection)
  - pre_burstlet_rate_hz   (separate run: thr=15, mp=0.05 — matches fitnessFunc_v2)

PDF pages
  Page 1: each rate feature vs T (linear x-axis, all grid Ts ticked, dashed
          T_full reference, red dotted vertical at closest-match T).
  Page 2: weighted "fitness" vs T per schema (schema_v2.py +
          multi_schema/schema_*.py). Fitness =
              Σ comp_w · metric_w · |m(T) − m(T_full)| / max_val
          over the schema's *rate-error* metrics. Lower = better.
  Page 3: focused single-schema view (default: schema_v2_equal.py) with the
          best T highlighted.

CLI outputs (siblings of --output):
  <output>.pdf                       (3-page)
  <output>.png                       (page 1)
  <output>_fitness.png               (page 2)
  <output>_focus_schema.png          (page 3)
  <output>.csv                       (rate features per T)
  <output>_fitness.csv               (per-schema fitness per T)
"""

import argparse
import csv
import importlib.util
import sys
from glob import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = Path("/pscratch/sd/k/ktub1999/networkSimulations")
sys.path.append(str(REPO_ROOT / "MEA_Analysis" / "IPNAnalysis"))
sys.path.append(str(REPO_ROOT / "RBS_network_models" / "_scripts"))

from parameter_free_burst_detector import compute_network_bursts  # noqa: E402
from create_experimental_target import classify_units, load_metrics  # noqa: E402


DEFAULT_T_GRID = [10, 20, 30, 45, 60, 90, 120, 180, 240]

FEATURES = [
    "exc_firing_rate_hz",
    "inh_firing_rate_hz",
    "mean_firing_rate_hz",
    "burstlet_rate_hz",
    "network_burst_rate_hz",
    "superburst_rate_hz",
    "pre_burstlet_rate_hz",
]

# Schema (component, metric) → feature key — only rate metrics are mapped here.
SCHEMA_RATE_MAP = {
    ("unit_metrics",   "firing_rate_error"):     "mean_firing_rate_hz",
    ("unit_metrics",   "firing_rate_error_exc"): "exc_firing_rate_hz",
    ("unit_metrics",   "firing_rate_error_inh"): "inh_firing_rate_hz",
    ("burstlets",      "burst_rate_error"):      "burstlet_rate_hz",
    ("network_bursts", "burst_rate_error"):      "network_burst_rate_hz",
    ("superbursts",    "burst_rate_error"):      "superburst_rate_hz",
    ("pre_burstlets",  "burst_rate_error"):      "pre_burstlet_rate_hz",
}

# Default schemas to compare on page 2.
SCHEMA_DIR = REPO_ROOT / "RBS_network_models" / "RBS_network_models" / "models" / \
    "CDKL5_E6D_T2_C1_05212024" / "DIV21_FxHET" / "fitness_schema"
DEFAULT_SCHEMAS = [str(SCHEMA_DIR / "schema_v2.py")] + sorted(
    glob(str(SCHEMA_DIR / "multi_schema" / "schema_*.py"))
)

# Schema rendered on its own page 3 (focused view + best-T decision).
DEFAULT_FOCUS_SCHEMA = str(SCHEMA_DIR / "schema_v2_equal.py")

# Production-matching burst detector params (see fitnessFunc_v2.py:861-867, 884-891).
# These are absolute (static) thresholds on ws_sharp — pass via base_threshold_static
# now that compute_network_bursts distinguishes static vs multiplier modes.
MAIN_BURST_KW = dict(base_threshold_static=40, min_burstlet_participation=0.05)
PRE_BURST_KW  = dict(base_threshold_static=15, min_burstlet_participation=0.05)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_spike_times(path):
    return np.load(path, allow_pickle=True).item()


def truncate(spike_times, T):
    out = {}
    for u, t in spike_times.items():
        arr = np.asarray(t)
        out[u] = arr[arr <= T]
    return out


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def firing_rates(truncated, cell_types, T):
    exc, inh, allu = [], [], []
    for u, t in truncated.items():
        rate = len(t) / T
        allu.append(rate)
        ct = cell_types.get(u)
        if ct == "excitatory":
            exc.append(rate)
        elif ct == "inhibitory":
            inh.append(rate)
    return (
        float(np.mean(exc)) if exc else np.nan,
        float(np.mean(inh)) if inh else np.nan,
        float(np.mean(allu)) if allu else np.nan,
    )


def _level_rate(nd, level):
    if "error" in nd:
        return 0.0
    return float(nd.get(level, {}).get("metrics", {}).get("rate", 0.0) or 0.0)


def burst_rates(truncated):
    """Return (burstlet, network, super, pre_burstlet) using production-matching params."""
    nd_main = compute_network_bursts(SpikeTimes=truncated, plot=False, verbose=False, **MAIN_BURST_KW)
    nd_pre  = compute_network_bursts(SpikeTimes=truncated, plot=False, verbose=False, **PRE_BURST_KW)
    return (
        _level_rate(nd_main, "burstlets"),
        _level_rate(nd_main, "network_bursts"),
        _level_rate(nd_main, "superbursts"),
        _level_rate(nd_pre,  "burstlets"),
    )


def compute_curve(spike_times, cell_types, T_grid):
    rows = []
    for T in T_grid:
        trunc = truncate(spike_times, T)
        if not any(len(t) for t in trunc.values()):
            rows.append({"T_s": T, **{k: 0.0 for k in FEATURES}})
            continue
        exc_fr, inh_fr, mean_fr = firing_rates(trunc, cell_types, T)
        bl_r, nb_r, sb_r, pb_r = burst_rates(trunc)
        rows.append({
            "T_s": float(T),
            "exc_firing_rate_hz": exc_fr,
            "inh_firing_rate_hz": inh_fr,
            "mean_firing_rate_hz": mean_fr,
            "burstlet_rate_hz": bl_r,
            "network_burst_rate_hz": nb_r,
            "superburst_rate_hz": sb_r,
            "pre_burstlet_rate_hz": pb_r,
        })
        print(f"  T={T:7.1f}s  exc={exc_fr:.3f}  inh={inh_fr:.3f}  mean={mean_fr:.3f}  "
              f"burstlet={bl_r:.4f}  network={nb_r:.4f}  super={sb_r:.4f}  preBL={pb_r:.4f}")
    return rows


def write_csv(rows, csv_path):
    fields = ["T_s"] + FEATURES
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


# ---------------------------------------------------------------------------
# Closest-match T (per feature, against T_full reference)
# ---------------------------------------------------------------------------

def closest_match(T, y, y_ref):
    if not np.isfinite(y_ref) or len(T) < 2:
        return None
    T_cand = T[:-1]
    y_cand = y[:-1]
    mask = np.isfinite(y_cand)
    if not mask.any():
        return None
    diffs = np.abs(y_cand[mask] - y_ref)
    i = int(np.argmin(diffs))
    T_best = float(T_cand[mask][i])
    y_best = float(y_cand[mask][i])
    rel = float(diffs[i] / abs(y_ref)) if y_ref != 0 else float("inf")
    return T_best, y_best, rel


# ---------------------------------------------------------------------------
# Schema loading + fitness
# ---------------------------------------------------------------------------

def load_schema(path):
    """Dynamically import a fitness-schema module and return its fit_schema dict."""
    spec = importlib.util.spec_from_file_location(f"schema_{Path(path).stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "fit_schema"):
        raise AttributeError(f"{path} has no fit_schema attribute")
    return mod.fit_schema


def schema_label(path):
    p = Path(path)
    return f"{p.parent.name}/{p.stem}" if p.parent.name == "multi_schema" else p.stem


def schema_rate_terms(fit_schema):
    """Yield (component, metric, comp_w, metric_w, max_val, feature_key) for rate-error
    metrics that map to one of our computed features and have non-zero combined weight."""
    for (comp, metric), feat in SCHEMA_RATE_MAP.items():
        if comp not in fit_schema:
            continue
        comp_w = float(fit_schema[comp].get("weight", 0.0))
        metric_block = fit_schema[comp].get("metrics", {}).get(metric)
        if metric_block is None:
            continue
        m_w = float(metric_block.get("weight", 0.0))
        if comp_w * m_w == 0.0:
            continue
        max_val = float(metric_block.get("max_val", 1.0)) or 1.0
        yield comp, metric, comp_w, m_w, max_val, feat


def schema_fitness_curve(fit_schema, rows):
    """For each T in rows, compute the schema's weighted absolute deviation from T_full
    summed over rate metrics and normalised by each metric's max_val."""
    terms = list(schema_rate_terms(fit_schema))
    if not terms:
        return None
    ref = rows[-1]
    fitness = []
    for r in rows:
        f = 0.0
        for _, _, comp_w, m_w, max_val, feat in terms:
            v_T = r.get(feat, np.nan)
            v_ref = ref.get(feat, np.nan)
            if not np.isfinite(v_T) or not np.isfinite(v_ref):
                continue
            f += comp_w * m_w * abs(v_T - v_ref) / max_val
        fitness.append(f)
    return fitness, terms


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_features_page(rows, label, pdf):
    T = np.array([r["T_s"] for r in rows], dtype=float)
    fig, axes = plt.subplots(4, 2, figsize=(11, 14))
    axes_flat = axes.flatten()
    summary_lines = []

    for ax, feat in zip(axes_flat[:len(FEATURES)], FEATURES):
        y = np.array([r.get(feat, np.nan) for r in rows], dtype=float)
        y_ref = y[-1]
        ax.plot(T, y, marker="o", lw=1.5)
        if np.isfinite(y_ref):
            ax.axhline(y_ref, ls="--", lw=1.0, color="gray",
                       label=f"T_full = {y_ref:.4g} Hz")
        match = closest_match(T, y, y_ref)
        title = feat
        if match is not None:
            T_best, y_best, rel = match
            ax.axvline(T_best, ls=":", lw=1.0, color="tab:red",
                       label=f"best T = {T_best:.0f} s (Δ {rel:.1%})")
            title = f"{feat}   |   best T = {T_best:.0f} s (Δ {rel:.1%})"
            summary_lines.append(f"{feat}: best T = {T_best:.0f} s, "
                                 f"value = {y_best:.4g} Hz, T_full = {y_ref:.4g} Hz "
                                 f"(Δ {rel:.1%})")
        ax.legend(loc="best", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("rate (Hz)")
        ax.set_xlabel("T (s)")
        ax.set_xticks(T)
        ax.set_xticklabels([f"{t:g}" for t in T], rotation=45, fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide the trailing unused axes (8 cells, 7 features).
    for ax in axes_flat[len(FEATURES):]:
        ax.axis("off")

    suptitle = f"Rate-feature convergence — {label}"
    if summary_lines:
        suptitle += "\nClosest-match T per feature (excludes T_full):\n" + "\n".join(summary_lines)
    fig.suptitle(suptitle, fontsize=11, ha="center")
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    pdf.savefig(fig)
    return fig, summary_lines


def plot_fitness_page(rows, schema_paths, label, pdf):
    T = np.array([r["T_s"] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(11, 8))
    fitness_per_schema = []  # list of (label, fitness_list, best_T, best_fit)
    n_schemas = len(schema_paths)
    cmap = plt.cm.tab10 if n_schemas <= 10 else plt.cm.tab20

    for i, sp in enumerate(schema_paths):
        slabel = schema_label(sp)
        try:
            fs = load_schema(sp)
        except Exception as e:
            print(f"  ! Failed to load {sp}: {e}")
            continue
        out = schema_fitness_curve(fs, rows)
        if out is None:
            print(f"  ! {slabel}: no rate metrics with non-zero weight; skipping.")
            continue
        fitness, terms = out
        # best T excludes T_full (last entry)
        f_arr = np.array(fitness, dtype=float)
        best_idx = int(np.argmin(f_arr[:-1])) if len(f_arr) > 1 else 0
        best_T = float(T[best_idx])
        best_fit = float(f_arr[best_idx])
        color = cmap(i % cmap.N)
        ax.plot(T, fitness, marker="o", lw=1.5, color=color,
                label=f"{slabel}  →  best T = {best_T:.0f} s (fit {best_fit:.4g})")
        ax.axvline(best_T, ls=":", lw=0.8, color=color, alpha=0.6)
        fitness_per_schema.append((slabel, fitness, best_T, best_fit, terms))

    ax.set_xlabel("window length T (s)")
    ax.set_ylabel("weighted absolute deviation from T_full   (lower = better)")
    ax.set_title(f"Schema fitness vs T — {label}", fontsize=12)
    ax.set_xticks(T)
    ax.set_xticklabels([f"{t:g}" for t in T], rotation=45, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    # Summary text under the plot listing best T per schema.
    summary_lines = ["Closest-match T per schema (excludes T_full):"]
    for slabel, _, best_T, best_fit, terms in fitness_per_schema:
        n_terms = len(terms)
        summary_lines.append(
            f"  {slabel:35s}  best T = {best_T:6.1f} s   "
            f"weighted Δ = {best_fit:.4g}   ({n_terms} rate terms)"
        )
    fig.text(0.02, 0.02, "\n".join(summary_lines),
             family="monospace", fontsize=8, va="bottom")

    fig.tight_layout(rect=[0, 0.02 + 0.02 * len(fitness_per_schema), 1, 1])
    pdf.savefig(fig)
    return fig, fitness_per_schema


def plot_single_schema_page(rows, schema_path, label, pdf):
    """One-page focused plot: fitness vs T for a single schema, with best T highlighted."""
    T = np.array([r["T_s"] for r in rows], dtype=float)
    slabel = schema_label(schema_path)
    try:
        fs = load_schema(schema_path)
    except Exception as e:
        print(f"  ! Failed to load {schema_path}: {e}")
        return None, None
    out = schema_fitness_curve(fs, rows)
    if out is None:
        print(f"  ! {slabel}: no rate metrics with non-zero weight; skipping focused page.")
        return None, None
    fitness, terms = out
    f_arr = np.array(fitness, dtype=float)
    best_idx = int(np.argmin(f_arr[:-1])) if len(f_arr) > 1 else 0
    best_T = float(T[best_idx])
    best_fit = float(f_arr[best_idx])

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.plot(T, fitness, marker="o", lw=1.8, color="tab:blue")
    ax.axvline(best_T, ls="--", lw=1.5, color="tab:red",
               label=f"best T = {best_T:.0f} s  (fit = {best_fit:.4g})")
    # mark every point with its fitness value
    for ti, fi in zip(T, fitness):
        ax.annotate(f"{fi:.3g}", xy=(ti, fi),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=7, color="dimgray")
    ax.set_xlabel("window length T (s)")
    ax.set_ylabel("weighted absolute deviation from T_full   (lower = better)")
    ax.set_title(f"Schema fitness vs T  —  {slabel}  —  {label}", fontsize=12)
    ax.set_xticks(T)
    ax.set_xticklabels([f"{t:g}" for t in T], rotation=45, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    # Term breakdown printed at the foot of the page.
    breakdown = [f"Schema: {slabel}",
                 f"Rate-error terms in fitness sum (component_weight × metric_weight / max_val):"]
    for comp, metric, comp_w, m_w, max_val, feat in terms:
        breakdown.append(
            f"  {comp:<16s} {metric:<24s} comp_w={comp_w:.2f} metric_w={m_w:.2f} "
            f"max_val={max_val:g}  →  feature `{feat}`"
        )
    breakdown.append("")
    breakdown.append(f"Lowest-error window: T = {best_T:.0f} s   weighted Δ = {best_fit:.4g}")
    fig.text(0.02, 0.02, "\n".join(breakdown),
             family="monospace", fontsize=8, va="bottom")
    fig.tight_layout(rect=[0, 0.02 + 0.025 * len(breakdown), 1, 1])
    pdf.savefig(fig)
    return fig, (slabel, fitness, best_T, best_fit, terms)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_grid(user_grid, T_full):
    grid = sorted({float(t) for t in user_grid if t > 0 and t <= T_full})
    if not grid or grid[-1] < T_full:
        grid.append(float(T_full))
    return grid


def write_fitness_csv(rows, fitness_per_schema, csv_path):
    T = [r["T_s"] for r in rows]
    fields = ["T_s"] + [s[0] for s in fitness_per_schema]
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for i, t in enumerate(T):
            row = [t] + [s[1][i] for s in fitness_per_schema]
            w.writerow(row)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spike_times", type=str, required=True)
    p.add_argument("--metrics", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--T_grid", type=float, nargs="+", default=DEFAULT_T_GRID)
    p.add_argument("--label", type=str, default=None)
    p.add_argument("--schemas", type=str, nargs="*", default=None,
                   help=f"Schema files. Default: schema_v2.py + multi_schema/schema_*.py "
                        f"(currently {len(DEFAULT_SCHEMAS)} files).")
    p.add_argument("--focus_schema", type=str, default=DEFAULT_FOCUS_SCHEMA,
                   help="Schema rendered on its own page 3 with best-T highlighted "
                        f"(default: {DEFAULT_FOCUS_SCHEMA}).")
    args = p.parse_args()

    out_pdf = Path(args.output)
    if out_pdf.suffix.lower() != ".pdf":
        out_pdf = out_pdf.with_suffix(".pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png         = out_pdf.with_suffix(".png")
    out_csv         = out_pdf.with_suffix(".csv")
    out_fitness_png = out_pdf.with_name(out_pdf.stem + "_fitness.png")
    out_fitness_csv = out_pdf.with_name(out_pdf.stem + "_fitness.csv")
    out_focus_png   = out_pdf.with_name(out_pdf.stem + "_focus_schema.png")

    label = args.label or Path(args.spike_times).parent.name or Path(args.spike_times).stem

    print(f"Loading spike_times: {args.spike_times}")
    spike_times = load_spike_times(args.spike_times)
    print(f"  {len(spike_times)} units")

    print(f"Loading metrics:     {args.metrics}")
    df = load_metrics(args.metrics)
    df = classify_units(df)
    cell_types = dict(zip(df["unit_id"], df["cell_type"]))
    n_exc = sum(1 for v in cell_types.values() if v == "excitatory")
    n_inh = sum(1 for v in cell_types.values() if v == "inhibitory")
    print(f"  classified: {n_exc} excitatory, {n_inh} inhibitory")

    T_full = max(
        (float(np.asarray(t).max()) for t in spike_times.values() if len(t)),
        default=0.0,
    )
    if T_full <= 0:
        raise SystemExit("No spikes found in spike_times — cannot build a window grid.")

    T_grid = build_grid(args.T_grid, T_full)
    print(f"T_full = {T_full:.2f} s; T_grid = {T_grid}")

    print("Sweeping rate features (this calls compute_network_bursts twice per T)...")
    rows = compute_curve(spike_times, cell_types, T_grid)

    print(f"Writing CSV: {out_csv}")
    write_csv(rows, out_csv)

    schema_paths = args.schemas if args.schemas else DEFAULT_SCHEMAS
    print(f"Schemas: {len(schema_paths)} file(s)")
    for sp in schema_paths:
        print(f"  - {sp}")

    print(f"Writing PDF: {out_pdf}")
    print(f"Writing PNG (page 1): {out_png}")
    print(f"Writing PNG (page 2): {out_fitness_png}")
    print(f"Writing PNG (page 3): {out_focus_png}")

    focus_summary = None
    with PdfPages(out_pdf) as pdf:
        fig1, summary = plot_features_page(rows, label, pdf)
        fig1.savefig(out_png, dpi=200)
        plt.close(fig1)

        fig2, fitness_per_schema = plot_fitness_page(rows, schema_paths, label, pdf)
        fig2.savefig(out_fitness_png, dpi=200)
        plt.close(fig2)

        if args.focus_schema:
            fig3, focus_summary = plot_single_schema_page(rows, args.focus_schema, label, pdf)
            if fig3 is not None:
                fig3.savefig(out_focus_png, dpi=200)
                plt.close(fig3)

    if fitness_per_schema:
        print(f"Writing fitness CSV: {out_fitness_csv}")
        write_fitness_csv(rows, fitness_per_schema, out_fitness_csv)

    if summary:
        print("Closest-match T per feature (excludes T_full):")
        for line in summary:
            print(f"  {line}")

    if fitness_per_schema:
        print("Closest-match T per schema (excludes T_full):")
        for slabel, _, best_T, best_fit, terms in fitness_per_schema:
            print(f"  {slabel:35s}  best T = {best_T:6.1f} s   "
                  f"weighted Δ = {best_fit:.4g}   ({len(terms)} rate terms)")

    if focus_summary is not None:
        slabel, _, best_T, best_fit, terms = focus_summary
        print(f"Focus schema decision: {slabel}  →  lowest-error T = {best_T:.0f} s   "
              f"weighted Δ = {best_fit:.4g}   ({len(terms)} rate terms)")

    print("Done.")


if __name__ == "__main__":
    main()
