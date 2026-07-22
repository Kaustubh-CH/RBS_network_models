#!/usr/bin/env python3
"""
compare_configs.py

Compare two simConfig JSON files by plotting all numeric parameters
side-by-side as bar charts. Output is saved as a multi-page PDF with
4 subplots per page.

Usage:
    python compare_configs.py <config1.json> <config2.json> [--output comparison.pdf]
"""

import json
import argparse
import os
import math
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def extract_numeric_params(sim_config):
    """Extract all scalar numeric parameters from a simConfig dict."""
    numeric_params = {}
    for key, value in sim_config.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric_params[key] = value
    return numeric_params


def make_short_label(path):
    """Create a short, readable label from a JSON file path.
    
    Extracts the trial name and a distinguishing parent directory segment.
    e.g. '.../CDKL5_seed_large_limited_tau_11_.../gen_81/trial_81_cfg.json'
      -> 'trial_81 (CDKL5_seed_large_limited_tau_11)'
    """
    basename = os.path.basename(path).replace('_cfg.json', '').replace('.json', '')
    parts = path.split('/')
    # Find a distinguishing directory name (look for one containing 'CDKL5' or 'seed')
    parent_hint = ''
    for part in parts:
        if 'CDKL5' in part or 'seed' in part:
            # Truncate long names
            parent_hint = part[:45] + ('...' if len(part) > 45 else '')
            break
    if parent_hint:
        nl = '\n'
        return f"{basename}{nl}({parent_hint})"
    return basename


def main():
    parser = argparse.ArgumentParser(
        description='Compare two simConfig JSON files with side-by-side bar plots.')
    parser.add_argument('config1', help='Path to first trial_*_cfg.json')
    parser.add_argument('config2', help='Path to second trial_*_cfg.json')
    parser.add_argument('--output', '-o', default=None,
                        help='Output PDF filename (default: config_comparison.pdf in CWD)')
    args = parser.parse_args()

    # --- Load JSON files ---
    with open(args.config1, 'r') as f:
        data1 = json.load(f)
    with open(args.config2, 'r') as f:
        data2 = json.load(f)

    cfg1 = data1.get('simConfig', data1)
    cfg2 = data2.get('simConfig', data2)

    # --- Extract numeric parameters ---
    params1 = extract_numeric_params(cfg1)
    params2 = extract_numeric_params(cfg2)

    # Union of all numeric keys, sorted
    all_keys = sorted(set(params1.keys()) | set(params2.keys()))

    if not all_keys:
        print("No numeric parameters found to compare.")
        return

    print(f"Found {len(all_keys)} numeric parameters to compare.")

    # --- Labels ---
    label1 = make_short_label(args.config1)
    label2 = make_short_label(args.config2)

    # --- Plotting ---
    SUBPLOTS_PER_PAGE = 4
    n_pages = math.ceil(len(all_keys) / SUBPLOTS_PER_PAGE)

    output_path = args.output or 'config_comparison.pdf'

    # Color palette
    color1 = '#4A90D9'  # steel blue
    color2 = '#E8744F'  # coral orange

    with PdfPages(output_path) as pdf:
        for page_idx in range(n_pages):
            start = page_idx * SUBPLOTS_PER_PAGE
            end = min(start + SUBPLOTS_PER_PAGE, len(all_keys))
            page_keys = all_keys[start:end]
            n_plots = len(page_keys)

            fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3.5 * n_plots))
            if n_plots == 1:
                axes = [axes]

            fig.suptitle(
                f'simConfig Parameter Comparison  (page {page_idx + 1}/{n_pages})',
                fontsize=14, fontweight='bold', y=0.99
            )

            for ax_idx, key in enumerate(page_keys):
                ax = axes[ax_idx]
                v1 = params1.get(key, 0)
                v2 = params2.get(key, 0)

                x = np.array([0, 1])
                bars = ax.bar(x, [v1, v2], width=0.5, color=[color1, color2],
                              edgecolor='#333333', linewidth=0.8)

                # Value annotations on top of bars
                for bar, val in zip(bars, [v1, v2]):
                    height = bar.get_height()
                    # Format: use scientific notation for large/small numbers
                    if abs(val) >= 1e4 or (abs(val) < 0.01 and val != 0):
                        txt = f'{val:.3e}'
                    else:
                        txt = f'{val:.4f}'
                    va = 'bottom' if height >= 0 else 'top'
                    ax.text(bar.get_x() + bar.get_width() / 2, height, txt,
                            ha='center', va=va, fontsize=9, fontweight='bold')

                ax.set_xticks(x)
                ax.set_xticklabels([label1, label2], fontsize=8)
                ax.set_ylabel(key, fontsize=11, fontweight='bold')
                ax.set_title(key, fontsize=12, fontweight='bold', pad=8)
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                # Mark if parameter is missing from one config
                if key not in params1:
                    ax.annotate('(missing)', xy=(0, 0), fontsize=8, color='red',
                                ha='center')
                if key not in params2:
                    ax.annotate('(missing)', xy=(1, 0), fontsize=8, color='red',
                                ha='center')

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved comparison PDF to: {os.path.abspath(output_path)}")
    print(f"  {n_pages} pages, {len(all_keys)} parameters compared.")


if __name__ == '__main__':
    main()
