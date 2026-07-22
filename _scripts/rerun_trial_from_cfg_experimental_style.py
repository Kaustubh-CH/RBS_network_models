"""
rerun_trial_from_cfg_experimental_style.py

Re-run a fitted trial network simulation from its saved `*_cfg.json` and produce
a single SVG with the raster plot on top and the network-activity panel directly
below it, matching the layout used in
RBS_network_models/plot_experimental_only.py.

Usage (MPI, matching batch.py):
    MPICH_GPU_SUPPORT_ENABLED=0 srun -n 32 \
        nrniv -python -mpi rerun_trial_from_cfg_experimental_style.py \
        <path/to/trial_xx_cfg.json> --duration-seconds 20 --output-dir ./op_20s/
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_RBS_ROOT = '/pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models'
sys.path.insert(0, _RBS_ROOT)
sys.path.insert(0, os.path.join(_RBS_ROOT, 'RBS_network_models'))
sys.path.insert(0, '/pscratch/sd/k/ktub1999/networkSimulations/MEA_Analysis/IPNAnalysis')

from netpyne import sim

# Reuse the exact helpers from plot_experimental_only's dependencies so the
# simulated plot matches the experimental layout pixel-for-pixel.
from fitnessFunc_v2 import (
    _run_burst_detector,
    _plot_raster,
    _plot_network_signal,
    _overlay_bursts,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def _is_master_rank() -> bool:
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank() == 0
    except Exception:
        return True


def find_netparams(cfg_path: Path) -> Path:
    batch_folder = cfg_path.parent.parent
    candidates = sorted(batch_folder.glob('*_netParams.py'))
    if not candidates:
        raise FileNotFoundError(
            f"Could not auto-detect netParams.py in {batch_folder}. "
            "Pass --netparams-path explicitly."
        )
    if len(candidates) > 1:
        logger.warning(f"Multiple netParams.py candidates: {candidates}. Using {candidates[0]}")
    return candidates[0]


def build_spike_data(sim_data, excit_set, inhib_set):
    """Convert NetPyNE simData into the {uid: np.array(times_s)} + cell_types dicts
    expected by _plot_raster / _run_burst_detector."""
    spkt_ms = np.asarray(sim_data.get('spkt', []), dtype=float)
    spkid = np.asarray(sim_data.get('spkid', []), dtype=int)

    spike_data = {}
    cell_types = {}
    if len(spkt_ms) and len(spkid):
        for gid in np.unique(spkid):
            gid_int = int(gid)
            spike_data[gid_int] = spkt_ms[spkid == gid] / 1000.0
            if gid_int in inhib_set:
                cell_types[gid_int] = 'inhibitory'
            else:
                cell_types[gid_int] = 'excitatory'
    return spike_data, cell_types


def compute_sim_burst_data(spike_data):
    """Same two-pass (main + pre) burst detection used by plot_experimental_only."""
    main = _run_burst_detector(
        spike_data,
        min_burstlet_participation=0.01,
        base_threshold_static=50,
    )
    pre = _run_burst_detector(
        spike_data,
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


def plot_simulated_experimental_style(spike_data, cell_types, duration_s,
                                      title_label, save_path):
    """Replicates the layout used in plot_experimental_only.plot_experimental."""
    burst = compute_sim_burst_data(spike_data)

    fig, (ax_raster, ax_net) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    _plot_raster(
        ax_raster, spike_data,
        title=f"Simulated Raster — {title_label}",
        cell_types=cell_types,
    )
    _overlay_bursts(ax_raster, burst['burst_events'])

    if burst['plot_data']:
        _plot_network_signal(ax_net, burst['plot_data'], title="Simulated Network Activity")
        _overlay_bursts(ax_net, burst['burst_events'])
    else:
        ax_net.text(0.5, 0.5, "No burst data", transform=ax_net.transAxes, ha='center')

    ax_net.set_xlabel("Time (s)")

    # --- Match the tick / legend overrides from plot_experimental_only ---
    ax_raster.set_yticks([0, 200, 400])
    ax_raster.set_xticks([0, duration_s])
    _, y_top = ax_raster.get_ylim()
    ax_raster.set_ylim(0, y_top)
    ax_raster.set_xlim(0, duration_s)

    # Cap network-activity y-axis at 100
    ax_net.set_yticks([10, 50])
    ax_net.set_xticks([0, duration_s])
    ax_net.set_ylim(0, 100)
    ax_net.set_xlim(0, duration_s)

    for ax in (ax_raster, ax_net):
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.08)
    fig.savefig(save_path, format='svg')
    plt.close(fig)
    logger.info(f"Saved experimental-style plot: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Re-run a fitted trial from its saved cfg JSON and plot '
                    'raster + network activity in the experimental layout.')
    parser.add_argument('cfg_json', help='Path to the saved trial *_cfg.json')
    parser.add_argument('--duration-seconds', type=float, default=None,
                        help='Override duration in seconds (network_cool_down added on top).')
    parser.add_argument('--netparams-path', default=None,
                        help='Path to netParams.py. Auto-detected from the batch folder if omitted.')
    parser.add_argument('--output-dir', default=None,
                        help='Directory for data.pkl + SVG plots.')
    parser.add_argument('--sim-label', default=None,
                        help='Override simLabel (prefix for saved files).')

    # nrniv passes extra argv entries in front of the script; trim to the script.
    for i, arg in enumerate(sys.argv):
        if arg.endswith('.py'):
            sys.argv = sys.argv[i:]
            break

    args = parser.parse_args()

    cfg_path = Path(args.cfg_json).expanduser().resolve()
    if not cfg_path.exists():
        logger.error(f'cfg JSON not found: {cfg_path}')
        sys.exit(1)

    netparams_path = (Path(args.netparams_path).expanduser().resolve()
                      if args.netparams_path else find_netparams(cfg_path))
    if not netparams_path.exists():
        logger.error(f'netParams.py not found: {netparams_path}')
        sys.exit(1)

    logger.info(f'cfg_json       = {cfg_path}')
    logger.info(f'netparams_path = {netparams_path}')

    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        simConfig, netParams = sim.readCmdLineArgs(
            simConfigDefault=str(cfg_path),
            netParamsDefault=str(netparams_path),
        )
    finally:
        sys.argv = saved_argv

    cool_down_s = float(getattr(simConfig, 'network_cool_down', 0.0) or 0.0)
    if args.duration_seconds is not None:
        new_sim_s = float(args.duration_seconds)
        simConfig.duration_seconds = new_sim_s
        simConfig.duration = (new_sim_s + cool_down_s) * 1e3
        logger.info(f'Overriding duration_seconds -> {new_sim_s}s (+{cool_down_s}s cool-down)')
    else:
        new_sim_s = float(getattr(simConfig, 'duration_seconds',
                                   simConfig.duration / 1e3 - cool_down_s))

    simConfig.printPopAvgRates = [100, simConfig.duration]
    simConfig.recordCells = []  # no voltage traces needed for this plot

    label_stem = cfg_path.stem.replace('_cfg', '')
    sim_label = args.sim_label or f'{label_stem}_rerun_{int(round(new_sim_s))}s'
    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        out_dir = cfg_path.parent / f'rerun_{label_stem}_{int(round(new_sim_s))}s'
    out_dir.mkdir(parents=True, exist_ok=True)

    simConfig.saveFolder = str(out_dir)
    simConfig.simLabel = sim_label
    simConfig.filename = sim_label
    simConfig.savePickle = True
    simConfig.saveJson = False
    logger.info(f'saveFolder = {simConfig.saveFolder}')
    logger.info(f'simLabel   = {sim_label}')

    sim.createSimulateAnalyze(simConfig=simConfig, netParams=netParams)

    if not _is_master_rank():
        return

    data_pkl = out_dir / f'{sim_label}_data.pkl'
    if not data_pkl.exists():
        alt = sorted(out_dir.glob(f'{sim_label}*_data.pkl'))
        if alt:
            data_pkl = alt[0]
    if not data_pkl.exists():
        logger.error(f'Expected data pickle not found in {out_dir}. Skipping plots.')
        return

    logger.info(f'Loading {data_pkl} for plotting...')
    with open(data_pkl, 'rb') as f:
        data = pickle.load(f)
    sim_data = data.get('simData', data)

    excit_set, inhib_set = set(), set()
    pops = data.get('net', {}).get('pops') or data.get('net', {}).get('allPops') or {}
    for pop_name, pop_info in pops.items():
        cell_gids = pop_info.get('cellGids', []) if hasattr(pop_info, 'get') \
            else getattr(pop_info, 'cellGids', [])
        target = inhib_set if str(pop_name) == 'I' else excit_set
        target.update(int(g) for g in cell_gids)
    logger.info(f'Pop assignments from net: E={len(excit_set)} gids, I={len(inhib_set)} gids')

    spike_data, cell_types = build_spike_data(sim_data, excit_set, inhib_set)
    logger.info(
        f'Built spike_data: {len(spike_data)} units, '
        f'{sum(len(s) for s in spike_data.values())} spikes'
    )

    out_svg = out_dir / f'{sim_label}_experimental_style.svg'
    plot_simulated_experimental_style(
        spike_data, cell_types, new_sim_s,
        title_label=label_stem, save_path=out_svg,
    )

    logger.info('Done.')
    logger.info(f'  data: {data_pkl}')
    logger.info(f'  plot: {out_svg}')


if __name__ == '__main__':
    main()

'''
Example invocation (32 ranks, full 20 s of trial_39):

srun -n 32 nrniv -mpi -python rerun_trial_from_cfg_experimental_style.py \
    /pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/CDKL5_Mar_25_seed_v3_large_02_w1_v3/batch_runs/batch_2026-03-25_spiking_only/gen_8/trial_8_cfg.json \
    --duration-seconds 20 \
    --output-dir ./op_20s_trial39/
'''
