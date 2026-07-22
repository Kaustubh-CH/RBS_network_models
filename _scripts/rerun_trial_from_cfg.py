"""
rerun_trial_from_cfg.py

Re-run a fitted trial network simulation from its saved `*_cfg.json`, keeping
the exact simConfig and netParams from the original run but allowing the
simulation duration to be changed. Produces:

    - <simLabel>_data.pkl  (written by NetPyNE when savePickle=True)
    - <simLabel>_raster.svg
    - <simLabel>_network_activity.svg
    - <simLabel>_waveforms.svg

Usage (single-process):
    python rerun_trial_from_cfg.py <path/to/trial_xx_cfg.json> \
        --duration-seconds 60 \
        --output-dir ./rerun_out

Usage (MPI, matching batch.py):
    MPICH_GPU_SUPPORT_ENABLED=0 srun -N 1 -n 128 \
        nrniv -python -mpi rerun_trial_from_cfg.py \
        <path/to/trial_xx_cfg.json> --duration-seconds 60

Notes:
    - The saved `*_cfg.json` is loaded verbatim via sim.readCmdLineArgs, so every
      evolved parameter, scale factor, and unit assignment is preserved.
    - The saved netParams.py is auto-detected from the batch folder (one level
      up from the trial's gen folder) if --netparams-path is not supplied.
    - recordCells is populated with a handful of E and I gids so waveforms can
      be plotted; everything else in simConfig is untouched.
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
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, '/pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models')

from netpyne import sim

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def _is_master_rank() -> bool:
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank() == 0
    except Exception:
        return True


def find_netparams(cfg_path: Path) -> Path:
    """Locate the batch-saved netParams.py.

    Batch layout:
        <batch_folder>/<batch_label>_netParams.py
        <batch_folder>/gen_<N>/trial_<N>_cfg.json
    """
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


def pick_record_cells(num_excite: int, num_inhib: int, n_each: int):
    """Return a recordCells spec capturing a few E and I gids.

    NetPyNE assigns sim gids in pop-creation order: pop 'E' gets gids
    [0..num_excite-1] and pop 'I' gets gids [num_excite..num_excite+num_inhib-1].
    cfg.excit_units / cfg.inhib_units are *experimental* unit IDs used only by
    netParams to look up cell positions — they are NOT simulation gids.
    """
    if n_each <= 0:
        return []
    e_sample = list(range(min(n_each, num_excite)))
    i_sample = list(range(num_excite, num_excite + min(n_each, num_inhib)))
    return [int(g) for g in e_sample + i_sample]


def plot_raster(spkt_s, spkid, excit_set, inhib_set, duration_s, out_path):
    fig, ax = plt.subplots(figsize=(14, 7))
    unique_ids = np.unique(spkid) if len(spkid) else np.array([])

    # Stable y ordering: E first, then I, then unknown.
    order = []
    for gid in unique_ids:
        if gid in excit_set:
            order.append((0, gid))
        elif gid in inhib_set:
            order.append((1, gid))
        else:
            order.append((2, gid))
    order.sort()
    id_to_y = {gid: y for y, (_, gid) in enumerate(order)}

    for gid, y in id_to_y.items():
        mask = spkid == gid
        if not np.any(mask):
            continue
        if gid in inhib_set:
            color = 'red'
        elif gid in excit_set:
            color = 'blue'
        else:
            color = 'gray'
        ax.plot(
            spkt_s[mask], np.full(mask.sum(), y),
            linestyle='None', marker='|', markersize=3,
            markeredgewidth=0.5, color=color, alpha=0.8, rasterized=True,
        )

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Unit (sorted: E, then I)')
    ax.set_xlim(0, duration_s)
    ax.set_ylim(-1, max(1, len(id_to_y)))
    ax.set_title(f'Spike raster ({duration_s:g} s)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(
        handles=[
            Line2D([0], [0], marker='|', color='blue', linestyle='None',
                   markersize=8, markeredgewidth=1.5, label='Excitatory'),
            Line2D([0], [0], marker='|', color='red', linestyle='None',
                   markersize=8, markeredgewidth=1.5, label='Inhibitory'),
        ],
        loc='upper right', fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, format='svg')
    plt.close(fig)


def plot_network_activity(spkt_s, duration_s, out_path, bin_size=0.01):
    if len(spkt_s) == 0:
        logger.warning('No spikes to plot for network activity')
        return
    n_bins = max(1, int(np.ceil(duration_s / bin_size)))
    bins = np.linspace(0, duration_s, n_bins + 1)
    counts, _ = np.histogram(spkt_s, bins=bins)
    centers = 0.5 * (bins[:-1] + bins[1:])

    window = max(1, int(0.5 / bin_size))  # 500 ms moving average
    kernel = np.ones(window) / window
    smoothed = np.convolve(counts, kernel, mode='same')

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(centers, counts, color='#B22222', lw=1.0, alpha=0.5, label='Spike count / 10 ms')
    ax.plot(centers, smoothed, color='tab:orange', lw=2.0, label='500 ms smoothed')
    baseline = float(np.mean(counts))
    threshold = baseline + float(np.std(counts))
    ax.axhline(baseline, color='#FF6600', ls='--', lw=1.2, alpha=0.8, label='Baseline')
    ax.axhline(threshold, color='#C0392B', ls='--', lw=1.2, alpha=0.8, label='Threshold')
    ax.set_xlim(0, duration_s)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Population spike count')
    ax.set_title('Network activity')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(out_path, format='svg')
    plt.close(fig)


def plot_waveforms(sim_data, duration_s, excit_set, inhib_set, out_path):
    """Plot soma voltage traces recorded by NetPyNE (one panel per recorded cell)."""
    traces_key = 'V_soma'
    traces = sim_data.get(traces_key)
    if traces is None:
        # Fallback: scan for any trace dict with 'cell_*' keys
        for k, v in sim_data.items():
            if isinstance(v, dict) and any(str(ck).startswith('cell_') for ck in v.keys()):
                traces_key = k
                traces = v
                break
    if not traces:
        logger.warning('No recorded voltage traces found in simData; skipping waveform plot')
        return

    t_ms = np.array(sim_data.get('t', []))
    if len(t_ms) == 0:
        logger.warning('No time vector in simData; skipping waveform plot')
        return
    t_s = t_ms / 1000.0

    cell_keys = sorted(traces.keys(), key=lambda s: int(str(s).split('_')[-1]) if str(s).split('_')[-1].isdigit() else 0)
    n = len(cell_keys)
    fig, axes = plt.subplots(n, 1, figsize=(14, max(2.0, 1.6 * n)), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, ck in zip(axes, cell_keys):
        trace = np.array(traces[ck])
        if len(trace) != len(t_s):
            L = min(len(trace), len(t_s))
            trace, tt = trace[:L], t_s[:L]
        else:
            tt = t_s
        # Parse gid from 'cell_<gid>'
        try:
            gid = int(str(ck).split('_')[-1])
        except ValueError:
            gid = None
        if gid is not None and gid in inhib_set:
            color, label = 'red', f'I cell gid={gid}'
        elif gid is not None and gid in excit_set:
            color, label = 'blue', f'E cell gid={gid}'
        else:
            color, label = 'gray', str(ck)
        ax.plot(tt, trace, color=color, lw=0.7)
        ax.set_ylabel('V (mV)')
        ax.set_title(label, fontsize=9, loc='left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[-1].set_xlabel('Time (s)')
    axes[-1].set_xlim(0, duration_s)
    fig.suptitle('Soma voltage traces', y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, format='svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Re-run a fitted trial from its saved cfg JSON.')
    parser.add_argument('cfg_json', help='Path to the saved trial *_cfg.json')
    parser.add_argument('--duration-seconds', type=float, default=None,
                        help='Override duration in seconds (network_cool_down from cfg is added on top).')
    parser.add_argument('--netparams-path', default=None,
                        help='Path to netParams.py. Auto-detected from the batch folder if omitted.')
    parser.add_argument('--output-dir', default=None,
                        help='Directory for data.pkl + SVG plots. Defaults to <cfg_dir>/rerun_<label>_<dur>s.')
    parser.add_argument('--sim-label', default=None,
                        help='Override simLabel (used as prefix for saved files).')
    parser.add_argument('--record-each', type=int, default=3,
                        help='Record soma V for this many E and this many I cells (default: 3 + 3).')
                        
    # Ensure sys.argv[0] is the script name when run via nrniv
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

    # Load simConfig + netParams exactly as the original run did.
    # sim.readCmdLineArgs also honors argv overrides (simConfig=, netParams=),
    # so we scrub argv to avoid accidental leakage from argparse.
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        simConfig, netParams = sim.readCmdLineArgs(
            simConfigDefault=str(cfg_path),
            netParamsDefault=str(netparams_path),
        )
    finally:
        sys.argv = saved_argv

    # Duration override (keep network_cool_down behavior from cfg.py intact).
    cool_down_s = float(getattr(simConfig, 'network_cool_down', 0.0) or 0.0)
    if args.duration_seconds is not None:
        new_sim_s = float(args.duration_seconds)
        simConfig.duration_seconds = new_sim_s
        simConfig.duration = (new_sim_s + cool_down_s) * 1e3  # ms
        logger.info(f'Overriding duration_seconds -> {new_sim_s}s (+{cool_down_s}s cool-down)')
    else:
        new_sim_s = float(getattr(simConfig, 'duration_seconds', simConfig.duration / 1e3 - cool_down_s))

    # printPopAvgRates window uses the new (longer/shorter) duration.
    simConfig.printPopAvgRates = [100, simConfig.duration]

    # Record a handful of E/I voltage traces for the waveform plot.
    num_excite = int(getattr(simConfig, 'num_excite', 0) or 0)
    num_inhib = int(getattr(simConfig, 'num_inhib', 0) or 0)
    simConfig.recordCells = pick_record_cells(num_excite, num_inhib, args.record_each)
    logger.info(
        f'recordCells ({len(simConfig.recordCells)} gids, num_excite={num_excite}, '
        f'num_inhib={num_inhib}): {simConfig.recordCells}'
    )

    # Output location.
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

    # Run the simulation. NetPyNE handles MPI internally; non-master ranks return early.
    sim.createSimulateAnalyze(simConfig=simConfig, netParams=netParams)

    if not _is_master_rank():
        return

    # Locate the pickle NetPyNE just wrote and load it for plotting.
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

    spkt_s = np.asarray(sim_data.get('spkt', []), dtype=float) / 1000.0
    spkid = np.asarray(sim_data.get('spkid', []), dtype=int)
    logger.info(f'Loaded {len(spkt_s)} spikes across {len(np.unique(spkid))} units')

    # Pop assignments come from the network, not cfg.excit_units/inhib_units —
    # those are experimental-unit IDs used by netParams to look up positions, not
    # simulation gids. NetPyNE assigns sim gids in pop-creation order (E first, I next).
    excit_set, inhib_set = set(), set()
    pops = data.get('net', {}).get('pops') or data.get('net', {}).get('allPops') or {}
    for pop_name, pop_info in pops.items():
        cell_gids = pop_info.get('cellGids', []) if hasattr(pop_info, 'get') \
            else getattr(pop_info, 'cellGids', [])
        target = inhib_set if str(pop_name) == 'I' else excit_set
        target.update(int(g) for g in cell_gids)
    logger.info(f'Pop assignments from net: E={len(excit_set)} gids, I={len(inhib_set)} gids')

    plot_raster(spkt_s, spkid, excit_set, inhib_set, new_sim_s,
                out_dir / f'{sim_label}_raster.svg')
    plot_network_activity(spkt_s, new_sim_s,
                          out_dir / f'{sim_label}_network_activity.svg')
    plot_waveforms(sim_data, new_sim_s, excit_set, inhib_set,
                   out_dir / f'{sim_label}_waveforms.svg')

    logger.info('Done.')
    logger.info(f'  data:       {data_pkl}')
    logger.info(f'  raster:     {out_dir / (sim_label + "_raster.svg")}')
    logger.info(f'  activity:   {out_dir / (sim_label + "_network_activity.svg")}')
    logger.info(f'  waveforms:  {out_dir / (sim_label + "_waveforms.svg")}')


if __name__ == '__main__':
    main()
'''
module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
module load cray-mpich
cd networkSimulations/RBS_network_models/_scripts
module load conda
conda activate preshifter
export LD_LIBRARY_PATH=$MPICH_DIR/ofi/gnu/$(gcc -dumpversion)/lib:$MPICH_DIR/gtl/lib:$LD_LIBRARY_PATH
export MPI_LIB_NRN_PATH=$(find $MPICH_DIR -name libmpi.so | head -1)
srun -n 32 nrniv -mpi -python rerun_trial_from_cfg.py /pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/CDKL5_seed_v4_02_w1_v1_multi_score/batch_runs_schema_v2/batch_2026-04-02_spiking_only/gen_135/trial_135_cfg.json --duration-seconds 20 --output-dir ./op_20s_better/
'''