#!/usr/bin/env python3
"""
plot_fitness_vs_gen.py

Reads all fitness JSON files from a batch run directory and plots fitness vs generation/trial.
Automatically detects whether the run used 'evol' (gen_X_cand_Y_fitness.json) or 
'optuna' (trial_X_fitness.json) naming conventions.

Usage:
    python plot_fitness_vs_gen.py <batch_dir> [--output plot.png] [--components] [--log] [--rolling N]
    
Examples:
    python plot_fitness_vs_gen.py /path/to/batch_runs/batch_2026-02-09_spiking_only
    python plot_fitness_vs_gen.py /path/to/batch_dir --components --log --rolling 10
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
from collections import defaultdict


def detect_run_mode(batch_dir):
    """Detect whether this is an evol run (gen_X_cand_Y) or optuna run (trial_X)."""
    gen_dirs = sorted(glob.glob(os.path.join(batch_dir, 'gen_*')))
    if not gen_dirs:
        raise FileNotFoundError(f"No gen_* directories found in {batch_dir}")
    
    # Check first gen dir for naming pattern
    first_gen = gen_dirs[0]
    trial_files = glob.glob(os.path.join(first_gen, 'trial_*_fitness.json'))
    cand_files = glob.glob(os.path.join(first_gen, 'gen_*_cand_*_fitness.json'))
    
    if trial_files:
        return 'optuna'
    elif cand_files:
        return 'evol'
    else:
        return 'optuna'
        raise FileNotFoundError(f"No fitness JSON files found in {first_gen}")


def load_fitness_data(batch_dir, mode=None):
    """
    Load all fitness JSONs from the batch directory.
    
    Returns:
        records: list of dicts, each with keys:
            'gen': int (generation number)
            'cand': int (candidate index, 0 for optuna)
            'trial': int (global trial number for optuna, same as gen)
            'fitness': float
            'components': dict (component-level scores if available)
            'weights': dict (component weights if available)
    """
    if mode is None:
        mode = detect_run_mode(batch_dir)
    
    print(f"Detected run mode: {mode}")
    
    gen_dirs = sorted(glob.glob(os.path.join(batch_dir, 'gen_*')))
    records = []
    missing = 0
    errors = 0
    
    for gen_dir in gen_dirs:
        gen_name = os.path.basename(gen_dir)
        # Extract generation number
        try:
            gen_num = int(gen_name.split('_')[1])
        except (IndexError, ValueError):
            continue
        
        if mode == 'optuna':
            # Pattern: gen_X/trial_X_fitness.json
            fitness_files = glob.glob(os.path.join(gen_dir, 'trial_*_fitness.json'))
            for fpath in fitness_files:
                fname = os.path.basename(fpath)
                try:
                    trial_num = int(fname.split('_')[1])
                except (IndexError, ValueError):
                    continue
                try:
                    with open(fpath, 'r') as f:
                        data = json.load(f)
                    record = {
                        'gen': gen_num,
                        'cand': 0,
                        'trial': trial_num,
                        'fitness': data.get('fitness', data.get('fit', None)),
                        'components': {},
                        'weights': data.get('weights', {}),
                    }
                    # Extract component scores
                    fd = data.get('fitness_dict', {})
                    for key, val in fd.items():
                        if isinstance(val, dict) and 'fit' in val:
                            record['components'][key] = val['fit']
                        elif isinstance(val, dict) and 'total_score' in val:
                            record['components'][key] = val['total_score']
                    
                    if record['fitness'] is not None:
                        records.append(record)
                except (json.JSONDecodeError, KeyError) as e:
                    errors += 1
                except FileNotFoundError:
                    missing += 1
                    
        elif mode == 'evol':
            # Pattern: gen_X/gen_X_cand_Y_fitness.json
            fitness_files = glob.glob(os.path.join(gen_dir, f'gen_{gen_num}_cand_*_fitness.json'))
            for fpath in sorted(fitness_files):
                fname = os.path.basename(fpath)
                try:
                    # gen_X_cand_Y_fitness.json
                    parts = fname.replace('_fitness.json', '').split('_')
                    cand_idx = int(parts[parts.index('cand') + 1])
                except (IndexError, ValueError):
                    continue
                try:
                    with open(fpath, 'r') as f:
                        data = json.load(f)
                    record = {
                        'gen': gen_num,
                        'cand': cand_idx,
                        'trial': None,  # filled later
                        'fitness': data.get('fitness', data.get('fit', None)),
                        'components': {},
                        'weights': data.get('weights', {}),
                    }
                    fd = data.get('fitness_dict', {})
                    for key, val in fd.items():
                        if isinstance(val, dict) and 'fit' in val:
                            record['components'][key] = val['fit']
                        elif isinstance(val, dict) and 'total_score' in val:
                            record['components'][key] = val['total_score']
                    
                    if record['fitness'] is not None:
                        records.append(record)
                except (json.JSONDecodeError, KeyError) as e:
                    errors += 1
                except FileNotFoundError:
                    missing += 1
    
    # Sort records
    records.sort(key=lambda r: (r['gen'], r['cand']))
    
    # Assign global trial numbers for evol mode
    if mode == 'evol':
        for i, r in enumerate(records):
            r['trial'] = i
    
    print(f"Loaded {len(records)} fitness records from {len(gen_dirs)} generation directories")
    if missing:
        print(f"  ({missing} missing files)")
    if errors:
        print(f"  ({errors} files with errors)")
    
    return records, mode


def compute_stats_per_gen(records):
    """Group records by generation and compute min/mean/max fitness per gen."""
    gen_data = defaultdict(list)
    for r in records:
        gen_data[r['gen']].append(r['fitness'])
    
    gens = sorted(gen_data.keys())
    stats = {
        'gens': gens,
        'min': [min(gen_data[g]) for g in gens],
        'mean': [np.mean(gen_data[g]) for g in gens],
        'max': [max(gen_data[g]) for g in gens],
        'n_cands': [len(gen_data[g]) for g in gens],
    }
    
    # Cumulative best
    cum_best = []
    best_so_far = float('inf')
    for g in gens:
        best_so_far = min(best_so_far, min(gen_data[g]))
        cum_best.append(best_so_far)
    stats['cum_best'] = cum_best
    
    return stats


def compute_component_stats(records):
    """Compute per-generation stats for each fitness component."""
    comp_names = set()
    for r in records:
        comp_names.update(r['components'].keys())
    
    # Filter out diagnostic/metadata keys
    skip = {'hierarchical_diagnostics', 'fit', 'quality_metrics'}
    comp_names = sorted(comp_names - skip)
    
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
            'min': [min(gen_data[g]) for g in gens],
        }
    
    return comp_stats


def rolling_average(values, window):
    """Compute rolling average with the given window size."""
    if window <= 1:
        return values
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(np.mean(values[start:i+1]))
    return result


def plot_fitness(records, mode, output_path, show_components=False, log_scale=False, rolling_window=1, y_cap=150):
    """
    Create fitness vs generation/trial plot.
    
    For evol mode: plots min/mean per generation with shaded range.
    For optuna mode: plots fitness per trial with cumulative best.
    """
    stats = compute_stats_per_gen(records)
    
    # Determine number of subplots
    n_plots = 1
    if show_components:
        comp_stats = compute_component_stats(records)
        if comp_stats:
            n_plots = 2
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 6 * n_plots), squeeze=False)
    ax = axes[0, 0]
    
    overflow_label_added = False

    if mode == 'evol':
        gens = stats['gens']
        mins = rolling_average(stats['min'], rolling_window)
        means = rolling_average(stats['mean'], rolling_window)
        maxs = stats['max']
        cum_best = stats['cum_best']

        mins_disp = np.minimum(mins, y_cap)
        means_disp = np.minimum(means, y_cap)
        maxs_disp = np.minimum(maxs, y_cap)
        cum_best_disp = np.minimum(cum_best, y_cap)
        
        ax.fill_between(gens, mins_disp, maxs_disp, alpha=0.15, color='steelblue', label='Range (min–max)')
        ax.plot(gens, means_disp, '-o', color='steelblue', markersize=3, linewidth=1.5, label='Mean fitness')
        ax.plot(gens, mins_disp, '-', color='darkgreen', linewidth=1.5, alpha=0.7, label='Best in gen')
        ax.plot(gens, cum_best_disp, '--', color='red', linewidth=2, label='Cumulative best')

        means_over = np.array(means) > y_cap
        mins_over = np.array(mins) > y_cap
        cum_over = np.array(cum_best) > y_cap
        if np.any(means_over):
            ax.scatter(np.array(gens)[means_over], np.full(np.sum(means_over), y_cap),
                       marker='^', s=45, color='steelblue', edgecolors='black',
                       linewidths=0.4, label=f'Values > {y_cap} (clipped)')
            overflow_label_added = True
        if np.any(mins_over):
            ax.scatter(np.array(gens)[mins_over], np.full(np.sum(mins_over), y_cap),
                       marker='^', s=45, color='darkgreen', edgecolors='black',
                       linewidths=0.4, label=None if overflow_label_added else f'Values > {y_cap} (clipped)')
            overflow_label_added = True
        if np.any(cum_over):
            ax.scatter(np.array(gens)[cum_over], np.full(np.sum(cum_over), y_cap),
                       marker='^', s=45, color='red', edgecolors='black',
                       linewidths=0.4, label=None if overflow_label_added else f'Values > {y_cap} (clipped)')
            overflow_label_added = True

        ax.set_xlabel('Generation', fontsize=13)
        ax.set_title(f'Fitness vs Generation  (pop_size={stats["n_cands"][0] if stats["n_cands"] else "?"})', fontsize=14)
        
    elif mode == 'optuna':
        trials = [r['trial'] for r in records]
        fitnesses = [r['fitness'] for r in records]
        
        # Cumulative best
        cum_best = []
        best = float('inf')
        for f in fitnesses:
            best = min(best, f)
            cum_best.append(best)
        
        # Rolling average
        smoothed = rolling_average(fitnesses, rolling_window)
        
        fitnesses_disp = np.minimum(fitnesses, y_cap)
        smoothed_disp = np.minimum(smoothed, y_cap)
        cum_best_disp = np.minimum(cum_best, y_cap)

        ax.scatter(trials, fitnesses_disp, s=20, alpha=0.5, color='steelblue', label='Trial fitness', zorder=2)
        if rolling_window > 1:
            ax.plot(trials, smoothed_disp, '-', color='steelblue', linewidth=1.5, alpha=0.8, 
                    label=f'Rolling avg (w={rolling_window})')
        ax.plot(trials, cum_best_disp, '-', color='red', linewidth=2, label='Cumulative best', zorder=3)

        fitness_over = np.array(fitnesses) > y_cap
        if np.any(fitness_over):
            ax.scatter(np.array(trials)[fitness_over], np.full(np.sum(fitness_over), y_cap),
                       marker='^', s=45, color='orange', edgecolors='black', linewidths=0.4,
                       label=f'Values > {y_cap} (clipped)', zorder=4)
        ax.set_xlabel('Trial', fontsize=13)
        ax.set_title(f'Fitness vs Trial (Optuna, {len(records)} trials)', fontsize=14)
    
    ax.set_ylabel('Fitness (lower = better)', fontsize=13)
    if log_scale:
        ax.set_yscale('log')
    ax.set_ylim(top=y_cap)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)
    
    # Annotate best
    best_record = min(records, key=lambda r: r['fitness'])
    label_key = 'trial' if mode == 'optuna' else 'gen'
    label_val = best_record[label_key]
    ax.annotate(
        f"Best: {best_record['fitness']:.3f}\n({label_key}={label_val})",
        xy=(label_val, best_record['fitness']),
        xytext=(30, 30), textcoords='offset points',
        fontsize=10, color='red',
        arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.8),
    )
    
    # Component breakdown plot
    if show_components and n_plots > 1:
        ax2 = axes[1, 0]
        colors = plt.cm.Set2(np.linspace(0, 1, len(comp_stats)))
        
        # Get weights from first record that has them
        weights = {}
        for r in records:
            if r['weights']:
                weights = r['weights']
                break
        
        for (comp_name, cdata), color in zip(comp_stats.items(), colors):
            w = weights.get(comp_name, '?')
            if isinstance(w, (int, float)):
                w_str = f'{w:.2f}'
            else:
                w_str = str(w)
            label = f'{comp_name} (w={w_str})'
            
            smoothed = rolling_average(cdata['mean'], rolling_window)
            ax2.plot(cdata['gens'], smoothed, '-o', color=color, markersize=2, 
                     linewidth=1.5, label=label)
        
        ax2.set_xlabel('Generation / Trial', fontsize=13)
        ax2.set_ylabel('Component Score (weighted)', fontsize=13)
        ax2.set_title('Fitness Components Over Time', fontsize=14)
        if log_scale:
            ax2.set_yscale('log')
        ax2.legend(fontsize=10, loc='upper right', ncol=2)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()


def print_summary(records, mode):
    """Print a summary table of optimization progress."""
    stats = compute_stats_per_gen(records)
    best = min(records, key=lambda r: r['fitness'])
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"  Mode:              {mode}")
    print(f"  Total records:     {len(records)}")
    print(f"  Generations/Trials: {len(stats['gens'])}")
    if mode == 'evol':
        print(f"  Pop size:          {stats['n_cands'][0] if stats['n_cands'] else '?'}")
    print(f"  Best fitness:      {best['fitness']:.4f}")
    label_key = 'trial' if mode == 'optuna' else 'gen'
    print(f"  Best {label_key}:       {best[label_key]}")
    if best['components']:
        print(f"  Components:")
        for k, v in sorted(best['components'].items()):
            w = best['weights'].get(k, 0)
            weighted = v * w if isinstance(w, (int, float)) else v
            print(f"    {k:20s}  raw={v:8.3f}  weight={w}  weighted={weighted:.3f}")
    print(f"  First fitness:     {records[0]['fitness']:.4f}")
    print(f"  Last fitness:      {records[-1]['fitness']:.4f}")
    improvement = records[0]['fitness'] - best['fitness']
    print(f"  Improvement:       {improvement:.4f} ({improvement/records[0]['fitness']*100:.1f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Plot fitness vs generation/trial from batch run.')
    parser.add_argument('batch_dir', type=str, help='Path to the batch run directory containing gen_* folders')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output plot path (default: <batch_dir>/fitness_vs_gen.png)')
    parser.add_argument('--components', '-c', action='store_true',
                        help='Show component-level fitness breakdown')
    parser.add_argument('--log', action='store_true',
                        help='Use log scale for y-axis')
    parser.add_argument('--rolling', '-r', type=int, default=1,
                        help='Rolling average window size (default: 1 = no smoothing)')
    parser.add_argument('--mode', choices=['evol', 'optuna'], default=None,
                        help='Force run mode (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    batch_dir = os.path.abspath(args.batch_dir)
    if not os.path.isdir(batch_dir):
        print(f"Error: {batch_dir} is not a directory")
        sys.exit(1)
    
    # Load data
    records, mode = load_fitness_data(batch_dir, mode=args.mode)
    if not records:
        print("Error: No fitness records found")
        sys.exit(1)
    
    # Print summary
    print_summary(records, mode)
    
    # Output path
    output_path = args.output or os.path.join(batch_dir, 'fitness_vs_gen.png')
    
    # Plot
    plot_fitness(records, mode, output_path, 
                 show_components=args.components,
                 log_scale=args.log,
                 rolling_window=args.rolling)


if __name__ == '__main__':
    main()

''' cd /pscratch/sd/k/ktub1999 && python3 /pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/RBS_network_models/plot_fitness_vs_gen.py \
/pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/CDKL5_seed_large_limited_tau_2_w0_focus_InhFR/batch_runs/batch_2026-04-02_spiking_only
 '''

'''
 cd /pscratch/sd/k/ktub1999 && for i in {1..5}; do python3 /pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/RBS_network_models/plot_fitness_vs_gen.py\
    "/pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/CDKL5_seed_large_limited_tau_2_w0_focus_InhFR/batch_runs_schema_v${i}/batch_2026-04-02_spiking_only/"; done

'''