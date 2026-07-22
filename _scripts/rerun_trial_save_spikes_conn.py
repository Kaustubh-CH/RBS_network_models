"""
rerun_trial_save_spikes_conn.py

Re-run a fitted trial network simulation from its saved `*_cfg.json` for an
arbitrary duration and, in addition to the full NetPyNE `_data.pkl`, save two
standalone artifacts:

    1. Spike times      -> <simLabel>_spikes.npz   (+ <simLabel>_spikes.csv)
    2. Connectivity     -> <simLabel>_connectivity.npz (+ <simLabel>_connectivity.csv)

The full `_data.pkl` already contains both (simData['spkt']/['spkid'] and
net['cells'][i]['conns']); these extra files are just convenient, self-contained
copies that load without unpickling the whole 60+ MB network object.

Usage (MPI, matching batch.py):
    MPICH_GPU_SUPPORT_ENABLED=0 srun -n 32 \
        nrniv -python -mpi rerun_trial_save_spikes_conn.py \
        <path/to/trial_xx_cfg.json> --duration-seconds 300 --output-dir ./op_300s/

Notes:
    - The saved `*_cfg.json` is loaded verbatim via sim.readCmdLineArgs, so every
      evolved parameter, scale factor, and unit assignment is preserved.
    - The saved netParams.py is auto-detected from the batch folder (one level up
      from the trial's gen folder) if --netparams-path is not supplied.
    - network_cool_down from the cfg is added on top of --duration-seconds, exactly
      as the original batch run did.
"""

import argparse
import csv
import logging
import os
import pickle
import sys
from pathlib import Path

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


def save_spike_times(sim_data, excit_set, inhib_set, out_stem: Path):
    """Save spike times as an .npz (arrays) and a .csv (one row per spike).

    spkt is stored in seconds. cell_type column is 'E'/'I'/'?'.
    """
    spkt_ms = np.asarray(sim_data.get('spkt', []), dtype=float)
    spkid = np.asarray(sim_data.get('spkid', []), dtype=int)
    spkt_s = spkt_ms / 1000.0

    def _ctype(g):
        if g in inhib_set:
            return 'I'
        if g in excit_set:
            return 'E'
        return '?'

    cell_types = np.array([_ctype(int(g)) for g in spkid])

    npz_path = out_stem.with_name(out_stem.name + '_spikes.npz')
    np.savez_compressed(
        npz_path,
        spkt_s=spkt_s,
        spkid=spkid,
        cell_type=cell_types,
        excit_gids=np.array(sorted(excit_set), dtype=int),
        inhib_gids=np.array(sorted(inhib_set), dtype=int),
    )

    csv_path = out_stem.with_name(out_stem.name + '_spikes.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['spike_time_s', 'gid', 'cell_type'])
        for t, g, c in zip(spkt_s, spkid, cell_types):
            w.writerow([f'{t:.6f}', int(g), c])

    logger.info(f'Saved {len(spkt_s)} spikes -> {npz_path}')
    logger.info(f'                        -> {csv_path}')
    return npz_path, csv_path


def save_connectivity(net, out_stem: Path):
    """Flatten net['cells'][i]['conns'] into an edge list and save as .npz + .csv.

    Each post-synaptic cell stores its incoming conns, so the edge is
    preGid (source) -> cell.gid (target).
    """
    cells = net.get('cells') if hasattr(net, 'get') else getattr(net, 'cells', [])

    pre_gid, post_gid, weight, delay = [], [], [], []
    synmech, label, sec, loc = [], [], [], []

    for cell in cells:
        post = cell.get('gid') if hasattr(cell, 'get') else getattr(cell, 'gid', None)
        conns = (cell.get('conns') if hasattr(cell, 'get') else getattr(cell, 'conns', None)) or []
        for cn in conns:
            g = cn.get('preGid')
            # Skip stim/NetStim sources that aren't integer cell gids.
            try:
                pre_gid.append(int(g))
            except (TypeError, ValueError):
                continue
            post_gid.append(int(post))
            weight.append(float(cn.get('weight', np.nan)))
            delay.append(float(cn.get('delay', np.nan)))
            synmech.append(str(cn.get('synMech', '')))
            label.append(str(cn.get('label', '')))
            sec.append(str(cn.get('sec', '')))
            loc.append(float(cn.get('loc', np.nan)))

    pre_gid = np.array(pre_gid, dtype=int)
    post_gid = np.array(post_gid, dtype=int)
    weight = np.array(weight, dtype=float)
    delay = np.array(delay, dtype=float)
    synmech = np.array(synmech)
    label = np.array(label)
    sec = np.array(sec)
    loc = np.array(loc, dtype=float)

    npz_path = out_stem.with_name(out_stem.name + '_connectivity.npz')
    np.savez_compressed(
        npz_path,
        pre_gid=pre_gid, post_gid=post_gid,
        weight=weight, delay=delay,
        synMech=synmech, label=label, sec=sec, loc=loc,
    )

    csv_path = out_stem.with_name(out_stem.name + '_connectivity.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pre_gid', 'post_gid', 'weight', 'delay', 'synMech', 'label', 'sec', 'loc'])
        for row in zip(pre_gid, post_gid, weight, delay, synmech, label, sec, loc):
            p, q, wt, dl, sm, lb, sc, lc = row
            w.writerow([int(p), int(q), f'{wt:.6g}', f'{dl:.6g}', sm, lb, sc, f'{lc:.4g}'])

    # Quick summary by label.
    uniq, counts = np.unique(label, return_counts=True)
    summary = dict(zip(uniq.tolist(), counts.tolist()))
    logger.info(f'Saved {len(pre_gid)} connections -> {npz_path}')
    logger.info(f'                               -> {csv_path}')
    logger.info(f'Connection breakdown by label: {summary}')
    return npz_path, csv_path


def main():
    parser = argparse.ArgumentParser(
        description='Re-run a fitted trial from its cfg JSON and save spike times '
                    '+ connectivity as standalone files (plus the full data.pkl).')
    parser.add_argument('cfg_json', help='Path to the saved trial *_cfg.json')
    parser.add_argument('--duration-seconds', type=float, default=None,
                        help='Override duration in seconds (network_cool_down added on top).')
    parser.add_argument('--netparams-path', default=None,
                        help='Path to netParams.py. Auto-detected from the batch folder if omitted.')
    parser.add_argument('--output-dir', default=None,
                        help='Directory for outputs. Defaults to <cfg_dir>/rerun_<label>_<dur>s.')
    parser.add_argument('--sim-label', default=None,
                        help='Override simLabel (prefix for saved files).')
    parser.add_argument('--fixed-dt', type=float, nargs='?', const=-1.0, default=None,
                        help='Disable CVode and integrate with a fixed timestep. Pass a value '
                             '(ms) to override dt, or bare --fixed-dt to keep the cfg dt. Use this '
                             'to avoid CVode h->hmin stalls / nrn_timeout aborts on long runs.')

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

    # Load simConfig + netParams exactly as the original run did.
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

    # Fixed-timestep override: disable CVode's variable stepping, which can drive
    # h -> hmin during stiff bursts and trigger nrn_timeout aborts on long runs.
    if args.fixed_dt is not None:
        simConfig.cvode_active = False
        if args.fixed_dt > 0:
            simConfig.dt = float(args.fixed_dt)
        logger.info(f'Fixed timestep enabled: cvode_active=False, dt={simConfig.dt} ms')

    simConfig.printPopAvgRates = [100, simConfig.duration]
    # Disable voltage-trace recording entirely. For long runs (e.g. 300 s) the
    # per-cell soma_voltage arrays (533 cells x duration/recordStep samples) reach
    # tens of GB and overflow NEURON's MPI all-to-all gather buffer
    # ("py_alltoall: cannot create std::vector larger than max_size()"). We only
    # need spikes + connectivity here, so drop traces.
    simConfig.recordCells = []
    simConfig.recordTraces = {}

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

    # Locate the pickle NetPyNE just wrote.
    data_pkl = out_dir / f'{sim_label}_data.pkl'
    if not data_pkl.exists():
        alt = sorted(out_dir.glob(f'{sim_label}*_data.pkl'))
        if alt:
            data_pkl = alt[0]
    if not data_pkl.exists():
        logger.error(f'Expected data pickle not found in {out_dir}. Skipping extraction.')
        return

    logger.info(f'Loading {data_pkl} for spike/connectivity extraction...')
    with open(data_pkl, 'rb') as f:
        data = pickle.load(f)
    sim_data = data.get('simData', data)
    net = data.get('net', {})

    # Pop assignments come from the network (NetPyNE assigns sim gids in
    # pop-creation order: E first, then I).
    excit_set, inhib_set = set(), set()
    pops = net.get('pops') or net.get('allPops') or {}
    for pop_name, pop_info in pops.items():
        cell_gids = pop_info.get('cellGids', []) if hasattr(pop_info, 'get') \
            else getattr(pop_info, 'cellGids', [])
        target = inhib_set if str(pop_name) == 'I' else excit_set
        target.update(int(g) for g in cell_gids)
    logger.info(f'Pop assignments from net: E={len(excit_set)} gids, I={len(inhib_set)} gids')

    out_stem = out_dir / sim_label
    save_spike_times(sim_data, excit_set, inhib_set, out_stem)
    save_connectivity(net, out_stem)

    logger.info('Done.')
    logger.info(f'  full data:     {data_pkl}')
    logger.info(f'  spikes:        {out_stem}_spikes.npz / .csv')
    logger.info(f'  connectivity:  {out_stem}_connectivity.npz / .csv')


if __name__ == '__main__':
    main()
