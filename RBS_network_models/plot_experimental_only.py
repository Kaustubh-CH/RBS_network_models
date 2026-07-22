"""
Plot experimental data only (raster + network activity), mirroring the
experimental panels of fitnessFunc_v2.plot_fitness_comparison.

Usage:
    python plot_experimental_only.py <h5_path> [-o <out.svg>]
"""

import argparse
import logging
import os
import sys
from typing import Dict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, '/pscratch/sd/k/ktub1999/networkSimulations/MEA_Analysis/IPNAnalysis')

from fitnessFunc_v2 import (
    load_experimental_h5,
    _run_burst_detector,
    _plot_raster,
    _plot_network_signal,
    _overlay_bursts,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def compute_exp_burst_data(exp_spike_data: Dict[int, np.ndarray]) -> Dict:
    """Run main + pre-burstlet detection for experimental spikes."""
    main = _run_burst_detector(
        exp_spike_data,
        min_burstlet_participation=0.05,
        base_threshold_static=40,
    )
    pre = _run_burst_detector(
        exp_spike_data,
        min_burstlet_participation=0.05,
        base_threshold_static=15,
    )

    main_pd = main.get('plot_data', {}) or {}
    pre_pd = pre.get('plot_data', {}) or {}

    plot_data = {
        **main_pd,
        'pre_burst_threshold': pre_pd.get('threshold', None),
        'pre_burst_peak_times': pre_pd.get('burst_peak_times', None),
        'pre_burst_peak_values': pre_pd.get('burst_peak_values', None),
    }

    burst_events = {
        level: main.get(level, {}).get('events', [])
        for level in ['burstlets', 'network_bursts', 'superbursts']
    }
    return {'plot_data': plot_data, 'burst_events': burst_events}


def plot_experimental(h5_path: str, save_path: str) -> None:
    exp_data = load_experimental_h5(h5_path)
    if exp_data['n_units'] == 0:
        raise RuntimeError(f"No units found in {h5_path}")

    logger.info(
        "Loaded %d units, %d total spikes, duration=%.2fs",
        exp_data['n_units'], exp_data['total_spikes'], exp_data['recording_duration'],
    )

    burst = compute_exp_burst_data(exp_data['spike_data'])

    exp_cell_types = {
        uid: exp_data['unit_attrs'].get(uid, {}).get('cell_type', 'excitatory')
        for uid in exp_data['unit_ids']
    }

    fig, (ax_raster, ax_net) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    _plot_raster(
        ax_raster, exp_data['spike_data'],
        title=f"Experimental Raster — {os.path.basename(h5_path)}",
        cell_types=exp_cell_types,
    )
    _overlay_bursts(ax_raster, burst['burst_events'])

    if burst['plot_data']:
        _plot_network_signal(ax_net, burst['plot_data'], title="Experimental Network Activity")
        _overlay_bursts(ax_net, burst['burst_events'])
    else:
        ax_net.text(0.5, 0.5, "No burst data", transform=ax_net.transAxes, ha='center')

    ax_net.set_xlabel("Time (s)")

    # --- Custom tick / legend overrides ---
    # Raster: y-ticks only at 0, 200, 400; x-ticks only at 0 and 20; both axes start at 0
    ax_raster.set_yticks([0, 200, 400])
    ax_raster.set_xticks([0, 20])
    _, y_top = ax_raster.get_ylim()
    ax_raster.set_ylim(0, y_top)
    _, x_right = ax_raster.get_xlim()
    ax_raster.set_xlim(0, x_right)
    # Network synchrony: y-ticks only at 10 and 50; x-ticks only at 0 and 20; x starts at 0
    ax_net.set_yticks([10, 50])
    ax_net.set_xticks([0, 20])
    _, x_right_n = ax_net.get_xlim()
    ax_net.set_xlim(0, x_right_n)

    # Drop legends from both panels
    for ax in (ax_raster, ax_net):
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.08)

    fig.savefig(save_path, format='svg')
    logger.info("Saved plot: %s", save_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot experimental raster + network activity as SVG")
    parser.add_argument(
        "h5_path",
        nargs='?',
        default="/pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/processed_experimental_targets/CDKL5_002_well000.h5",
        help="Path to experimental_features.h5 file",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output SVG path (default: <h5_basename>_experimental_plot.svg in CWD)",
    )
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.h5_path))[0]
        args.output = f"{base}_experimental_plot.svg"

    plot_experimental(args.h5_path, args.output)


if __name__ == "__main__":
    main()
