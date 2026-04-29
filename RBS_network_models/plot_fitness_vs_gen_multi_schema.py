#!/usr/bin/env python3
"""
plot_fitness_vs_gen_multi_schema.py

Plots fitness-vs-generation for EACH batch_runs_schema_vN (N = 1..5) found
inside a root simulation directory, writing all pages into a single PDF.

Layout per page (one page per schema):
  - Top panel    : total fitness vs generation / trial  (min / mean / cum-best)
  - Bottom panel : per-component fitness breakdown  (optional, --components)

Usage
-----
    python plot_fitness_vs_gen_multi_schema.py <root_dir> [options]

    <root_dir>   Directory that contains batch_runs_schema_v1 … batch_runs_schema_v5
                 (e.g. /path/to/CDKL5_seed_v3_large_v2_02_w1_v2)

Options
-------
    --output   / -o   Output PDF path  (default: <root_dir>/fitness_multi_schema.pdf)
    --schemas  / -s   Comma-separated schema indices to plot  (default: 1,2,3,4,5)
    --batch           Sub-directory name pattern inside each schema dir to look for
                      (default: auto-detect the newest batch_* subdirectory)
    --components / -c  Show per-component breakdown panel on each page
    --log             Log scale for y-axis
    --rolling  / -r   Rolling-average window (default: 1 = no smoothing)
    --ycap            Upper y-axis clipping value (default: 150)
    --mode            Force run mode: 'evol' or 'optuna' (auto-detected by default)

Examples
--------
    python plot_fitness_vs_gen_multi_schema.py \\
        /pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/CDKL5_seed_v3_large_v2_02_w1_v2

    python plot_fitness_vs_gen_multi_schema.py \\
        /pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/CDKL5_seed_v3_large_v2_02_w1_v2 \\
        --components --rolling 5 --ycap 200
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict


# ---------------------------------------------------------------------------
# Helpers re-used from plot_fitness_vs_gen.py
# ---------------------------------------------------------------------------

def detect_run_mode(batch_dir):
    """Detect whether this is an evol run (gen_X_cand_Y) or optuna run (trial_X)."""
    gen_dirs = sorted(glob.glob(os.path.join(batch_dir, 'gen_*')))
    if not gen_dirs:
        raise FileNotFoundError(f"No gen_* directories found in {batch_dir}")

    first_gen = gen_dirs[0]
    trial_files = glob.glob(os.path.join(first_gen, 'trial_*_fitness.json'))
    cand_files  = glob.glob(os.path.join(first_gen, 'gen_*_cand_*_fitness.json'))

    if trial_files:
        return 'optuna'
    elif cand_files:
        return 'evol'
    else:
        return 'optuna'


def load_fitness_data(batch_dir, mode=None):
    """
    Load all fitness JSONs from the batch directory.

    Returns
    -------
    records : list[dict]  — each with keys gen, cand, trial, fitness, components, weights
    mode    : str         — 'evol' or 'optuna'
    """
    if mode is None:
        mode = detect_run_mode(batch_dir)

    gen_dirs = sorted(glob.glob(os.path.join(batch_dir, 'gen_*')))
    records = []
    missing = errors = 0

    for gen_dir in gen_dirs:
        gen_name = os.path.basename(gen_dir)
        try:
            gen_num = int(gen_name.split('_')[1])
        except (IndexError, ValueError):
            continue

        if mode == 'optuna':
            fitness_files = glob.glob(os.path.join(gen_dir, 'trial_*_fitness.json'))
            for fpath in fitness_files:
                fname = os.path.basename(fpath)
                try:
                    trial_num = int(fname.split('_')[1])
                except (IndexError, ValueError):
                    continue
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                    rec = {
                        'gen': gen_num, 'cand': 0, 'trial': trial_num,
                        'fitness': data.get('fitness', data.get('fit')),
                        'components': {}, 'weights': data.get('weights', {}),
                    }
                    for key, val in data.get('fitness_dict', {}).items():
                        if isinstance(val, dict) and 'fit' in val:
                            rec['components'][key] = val['fit']
                        elif isinstance(val, dict) and 'total_score' in val:
                            rec['components'][key] = val['total_score']
                    if rec['fitness'] is not None:
                        records.append(rec)
                except (json.JSONDecodeError, KeyError):
                    errors += 1
                except FileNotFoundError:
                    missing += 1

        elif mode == 'evol':
            fitness_files = glob.glob(
                os.path.join(gen_dir, f'gen_{gen_num}_cand_*_fitness.json'))
            for fpath in sorted(fitness_files):
                fname = os.path.basename(fpath)
                try:
                    parts = fname.replace('_fitness.json', '').split('_')
                    cand_idx = int(parts[parts.index('cand') + 1])
                except (IndexError, ValueError):
                    continue
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                    rec = {
                        'gen': gen_num, 'cand': cand_idx, 'trial': None,
                        'fitness': data.get('fitness', data.get('fit')),
                        'components': {}, 'weights': data.get('weights', {}),
                    }
                    for key, val in data.get('fitness_dict', {}).items():
                        if isinstance(val, dict) and 'fit' in val:
                            rec['components'][key] = val['fit']
                        elif isinstance(val, dict) and 'total_score' in val:
                            rec['components'][key] = val['total_score']
                    if rec['fitness'] is not None:
                        records.append(rec)
                except (json.JSONDecodeError, KeyError):
                    errors += 1
                except FileNotFoundError:
                    missing += 1

    records.sort(key=lambda r: (r['gen'], r['cand']))
    if mode == 'evol':
        for i, r in enumerate(records):
            r['trial'] = i

    if missing:
        print(f"    ({missing} missing files)")
    if errors:
        print(f"    ({errors} files with errors)")

    return records, mode


def compute_stats_per_gen(records):
    gen_data = defaultdict(list)
    for r in records:
        gen_data[r['gen']].append(r['fitness'])
    gens = sorted(gen_data.keys())
    cum_best, best_so_far = [], float('inf')
    for g in gens:
        best_so_far = min(best_so_far, min(gen_data[g]))
        cum_best.append(best_so_far)
    return {
        'gens':     gens,
        'min':      [min(gen_data[g]) for g in gens],
        'mean':     [np.mean(gen_data[g]) for g in gens],
        'max':      [max(gen_data[g]) for g in gens],
        'n_cands':  [len(gen_data[g]) for g in gens],
        'cum_best': cum_best,
    }


def compute_component_stats(records):
    skip = {'hierarchical_diagnostics', 'fit', 'quality_metrics'}
    comp_names = sorted(
        {k for r in records for k in r['components']} - skip
    )
    comp_stats = {}
    for comp in comp_names:
        gen_data = defaultdict(list)
        for r in records:
            if comp in r['components']:
                gen_data[r['gen']].append(r['components'][comp])
        gens = sorted(gen_data.keys())
        comp_stats[comp] = {
            'gens': gens,
            'mean': [np.mean(gen_data[g]) for g in gens],
            'min':  [min(gen_data[g]) for g in gens],
        }
    return comp_stats


def rolling_average(values, window):
    if window <= 1:
        return list(values)
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(np.mean(values[start:i+1]))
    return result


# ---------------------------------------------------------------------------
# Per-page plot (one schema)
# ---------------------------------------------------------------------------

SCHEMA_COLORS = {
    1: '#4C72B0',
    2: '#DD8452',
    3: '#55A868',
    4: '#C44E52',
    5: '#8172B3',
}

def plot_schema_page(
    pdf: PdfPages,
    schema_idx: int,
    schema_label: str,
    records: list,
    mode: str,
    batch_dir: str,
    show_components: bool = False,
    log_scale: bool = False,
    rolling_window: int = 1,
    y_cap: float = 150.0,
):
    """Render one page for a single schema and add it to the PDF."""
    stats = compute_stats_per_gen(records)
    comp_stats = compute_component_stats(records) if show_components else {}
    n_plots = 2 if (show_components and comp_stats) else 1

    fig_height = 6 * n_plots + 1.5  # extra room for suptitle
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, fig_height), squeeze=False)
    color = SCHEMA_COLORS.get(schema_idx, '#333333')

    # ----------------------------------------------------------------
    # Suptitle — schema label + batch dir
    # ----------------------------------------------------------------
    fig.suptitle(
        f'{schema_label}\n{os.path.basename(batch_dir)}',
        fontsize=13, fontweight='bold', y=0.99,
    )

    # ----------------------------------------------------------------
    # Top panel: total fitness
    # ----------------------------------------------------------------
    ax = axes[0, 0]
    overflow_label_added = False

    if mode == 'evol':
        gens      = stats['gens']
        mins      = rolling_average(stats['min'], rolling_window)
        means     = rolling_average(stats['mean'], rolling_window)
        maxs      = stats['max']
        cum_best  = stats['cum_best']

        ax.fill_between(gens, np.minimum(mins, y_cap), np.minimum(maxs, y_cap),
                        alpha=0.15, color=color, label='Range (min–max)')
        ax.plot(gens, np.minimum(means, y_cap), '-o', color=color,
                markersize=3, linewidth=1.5, label='Mean fitness')
        ax.plot(gens, np.minimum(mins, y_cap), '-', color='darkgreen',
                linewidth=1.5, alpha=0.7, label='Best in gen')
        ax.plot(gens, np.minimum(cum_best, y_cap), '--', color='red',
                linewidth=2, label='Cumulative best')

        for vals, clr in [(means, color), (mins, 'darkgreen'), (cum_best, 'red')]:
            over = np.array(vals) > y_cap
            if np.any(over):
                lbl = f'Values > {y_cap} (clipped)' if not overflow_label_added else None
                ax.scatter(np.array(gens)[over], np.full(np.sum(over), y_cap),
                           marker='^', s=45, color=clr,
                           edgecolors='black', linewidths=0.4, label=lbl)
                overflow_label_added = True

        ax.set_xlabel('Generation', fontsize=12)
        pop_size = stats['n_cands'][0] if stats['n_cands'] else '?'
        ax.set_title(f'Fitness vs Generation  (pop_size={pop_size})', fontsize=13)

    elif mode == 'optuna':
        trials   = [r['trial'] for r in records]
        fitnesses = [r['fitness'] for r in records]
        cum_best = []
        best = float('inf')
        for fv in fitnesses:
            best = min(best, fv)
            cum_best.append(best)
        smoothed = rolling_average(fitnesses, rolling_window)

        ax.scatter(trials, np.minimum(fitnesses, y_cap), s=20, alpha=0.5,
                   color=color, label='Trial fitness', zorder=2)
        if rolling_window > 1:
            ax.plot(trials, np.minimum(smoothed, y_cap), '-', color=color,
                    linewidth=1.5, alpha=0.8, label=f'Rolling avg (w={rolling_window})')
        ax.plot(trials, np.minimum(cum_best, y_cap), '-', color='red',
                linewidth=2, label='Cumulative best', zorder=3)

        over = np.array(fitnesses) > y_cap
        if np.any(over):
            ax.scatter(np.array(trials)[over], np.full(np.sum(over), y_cap),
                       marker='^', s=45, color='orange',
                       edgecolors='black', linewidths=0.4,
                       label=f'Values > {y_cap} (clipped)', zorder=4)

        ax.set_xlabel('Trial', fontsize=12)
        ax.set_title(f'Fitness vs Trial  (Optuna, {len(records)} trials)', fontsize=13)

    ax.set_ylabel('Fitness (lower = better)', fontsize=12)
    if log_scale:
        ax.set_yscale('log')
    ax.set_ylim(top=y_cap)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)

    # Annotate best record
    best_rec   = min(records, key=lambda r: r['fitness'])
    label_key  = 'trial' if mode == 'optuna' else 'gen'
    label_val  = best_rec[label_key]
    xy_y       = min(best_rec['fitness'], y_cap)
    ax.annotate(
        f"Best: {best_rec['fitness']:.3f}\n({label_key}={label_val})",
        xy=(label_val, xy_y),
        xytext=(30, 30), textcoords='offset points',
        fontsize=9, color='red',
        arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='red', alpha=0.8),
    )

    # Add summary text box in top-left corner of fitness panel
    n_gens = len(stats['gens'])
    cum_best_final = stats['cum_best'][-1] if stats['cum_best'] else float('nan')
    summary_txt = (
        f"Schema v{schema_idx}\n"
        f"Records : {len(records)}\n"
        f"Gens    : {n_gens}\n"
        f"Best    : {cum_best_final:.3f}"
    )
    ax.text(0.01, 0.97, summary_txt, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='gray'))

    # ----------------------------------------------------------------
    # Bottom panel: component breakdown
    # ----------------------------------------------------------------
    if show_components and comp_stats:
        ax2 = axes[1, 0]
        palette = plt.cm.tab10(np.linspace(0, 1, max(len(comp_stats), 1)))

        # Weights from first record that has them
        weights = next((r['weights'] for r in records if r['weights']), {})

        for (comp_name, cdata), clr in zip(comp_stats.items(), palette):
            w = weights.get(comp_name, '?')
            w_str = f'{w:.2f}' if isinstance(w, (int, float)) else str(w)
            label = f'{comp_name} (w={w_str})'
            smoothed = rolling_average(cdata['mean'], rolling_window)
            ax2.plot(cdata['gens'], smoothed, '-o', color=clr,
                     markersize=2, linewidth=1.5, label=label)

        ax2.set_xlabel('Generation / Trial', fontsize=12)
        ax2.set_ylabel('Component Score', fontsize=12)
        ax2.set_title('Fitness Components Over Time', fontsize=13)
        if log_scale:
            ax2.set_yscale('log')
        ax2.legend(fontsize=9, loc='upper right', ncol=2)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Directory resolution
# ---------------------------------------------------------------------------

def find_batch_subdir(schema_dir: str, batch_pattern: str = None) -> str:
    """
    Return the batch subdirectory inside schema_dir.
    If batch_pattern is given, look for that name directly.
    Otherwise auto-pick the alphabetically-latest batch_* folder.
    """
    if batch_pattern:
        candidate = os.path.join(schema_dir, batch_pattern)
        if os.path.isdir(candidate):
            return candidate
        # Maybe it's an exact full path
        if os.path.isdir(batch_pattern):
            return batch_pattern
        raise FileNotFoundError(
            f"Batch subdirectory '{batch_pattern}' not found in {schema_dir}")

    # Auto-detect: pick latest batch_* dir
    candidates = sorted(glob.glob(os.path.join(schema_dir, 'batch_*')))
    if not candidates:
        raise FileNotFoundError(f"No batch_* subdirectory found in {schema_dir}")
    return candidates[-1]  # newest by name (YYYY-MM-DD sorts correctly)


# ---------------------------------------------------------------------------
# Summary page
# ---------------------------------------------------------------------------

def plot_summary_page(pdf: PdfPages, all_schema_data: list, y_cap: float):
    """
    First page of the PDF: overlay cumulative-best curves for all schemas.
    all_schema_data: list of (schema_idx, schema_label, records, mode, color)
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle('Cumulative Best Fitness — All Schemas', fontsize=14, fontweight='bold')

    for schema_idx, schema_label, records, mode, color in all_schema_data:
        if not records:
            continue
        stats = compute_stats_per_gen(records)
        gens      = stats['gens']
        cum_best  = np.minimum(stats['cum_best'], y_cap)
        ax.plot(gens, cum_best, '-o', color=color, markersize=3,
                linewidth=2, label=schema_label)
        # Annotate final value
        ax.annotate(
            f"{stats['cum_best'][-1]:.2f}", xy=(gens[-1], cum_best[-1]),
            xytext=(5, 0), textcoords='offset points',
            fontsize=8, color=color, va='center',
        )

    ax.set_xlabel('Generation / Trial', fontsize=12)
    ax.set_ylabel('Cumulative Best Fitness (lower = better)', fontsize=12)
    ax.set_ylim(top=y_cap)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Plot fitness-vs-generation for all batch_runs_schema_vN into a PDF.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('root_dir', type=str,
                        help='Root directory containing batch_runs_schema_v1..5')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output PDF path (default: <root_dir>/fitness_multi_schema.pdf)')
    parser.add_argument('--schemas', '-s', type=str, default='1,2,3,4,5',
                        help='Comma-separated schema indices to include (default: 1,2,3,4,5)')
    parser.add_argument('--batch', type=str, default=None,
                        help='Batch sub-directory name (auto-detected if omitted)')
    parser.add_argument('--components', '-c', action='store_true',
                        help='Show per-component breakdown panel')
    parser.add_argument('--log', action='store_true',
                        help='Log scale for y-axis')
    parser.add_argument('--rolling', '-r', type=int, default=1,
                        help='Rolling-average window size (default: 1)')
    parser.add_argument('--ycap', type=float, default=150.0,
                        help='Upper y clipping value (default: 150)')
    parser.add_argument('--mode', choices=['evol', 'optuna'], default=None,
                        help='Force run mode (auto-detected if omitted)')

    args = parser.parse_args()

    root_dir = os.path.abspath(args.root_dir)
    if not os.path.isdir(root_dir):
        print(f"Error: {root_dir} is not a directory")
        sys.exit(1)

    schema_indices = []
    for tok in args.schemas.split(','):
        tok = tok.strip()
        if tok:
            schema_indices.append(int(tok))

    output_path = args.output or os.path.join(root_dir, 'fitness_multi_schema.pdf')

    # ------------------------------------------------------------------
    # Collect data for all schemas
    # ------------------------------------------------------------------
    all_schema_data = []   # (idx, label, records, mode, color)

    for idx in schema_indices:
        schema_dir = os.path.join(root_dir, f'batch_runs_schema_v{idx}')
        if not os.path.isdir(schema_dir):
            print(f"[schema v{idx}] Directory not found: {schema_dir}  — skipping")
            continue

        try:
            batch_dir = find_batch_subdir(schema_dir, args.batch)
        except FileNotFoundError as e:
            print(f"[schema v{idx}] {e}  — skipping")
            continue

        print(f"\n[schema v{idx}] Loading from: {batch_dir}")
        try:
            records, mode = load_fitness_data(batch_dir, mode=args.mode)
        except FileNotFoundError as e:
            print(f"[schema v{idx}] {e}  — skipping")
            continue

        if not records:
            print(f"[schema v{idx}] No fitness records found  — skipping")
            continue

        best = min(records, key=lambda r: r['fitness'])
        label_key = 'trial' if mode == 'optuna' else 'gen'
        print(f"[schema v{idx}]   {len(records)} records | mode={mode} | "
              f"best={best['fitness']:.4f} at {label_key}={best[label_key]}")

        color = SCHEMA_COLORS.get(idx, '#555555')
        label = f'Schema v{idx}'
        all_schema_data.append((idx, label, records, mode, color))

    if not all_schema_data:
        print("\nNo valid schema data found. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Write PDF
    # ------------------------------------------------------------------
    print(f"\nWriting PDF → {output_path}")
    with PdfPages(output_path) as pdf:
        # Page 0: summary overlay of all cumulative-best curves
        plot_summary_page(pdf, all_schema_data, y_cap=args.ycap)

        # One page per schema
        for idx, schema_label, records, mode, color in all_schema_data:
            schema_dir = os.path.join(root_dir, f'batch_runs_schema_v{idx}')
            try:
                batch_dir = find_batch_subdir(schema_dir, args.batch)
            except FileNotFoundError:
                batch_dir = schema_dir

            print(f"  Adding page: {schema_label}")
            plot_schema_page(
                pdf=pdf,
                schema_idx=idx,
                schema_label=schema_label,
                records=records,
                mode=mode,
                batch_dir=batch_dir,
                show_components=args.components,
                log_scale=args.log,
                rolling_window=args.rolling,
                y_cap=args.ycap,
            )

        # PDF metadata
        d = pdf.infodict()
        d['Title']   = 'Fitness vs Generation — Multi Schema'
        d['Subject'] = f'Schemas: {", ".join(str(i) for i,*_ in all_schema_data)}'

    print(f"\nDone. PDF saved to: {output_path}")
    print(f"Total pages: {1 + len(all_schema_data)}  (summary + one per schema)")


if __name__ == '__main__':
    main()

'''module load conda && conda run -n preshifter python3 \
  /pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/RBS_network_models/plot_fitness_vs_gen_multi_schema.py \
/pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/CDKL5_large_limited_tau_02_w1_v1_multi_score/

'''