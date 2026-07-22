"""
plot_spikes_conn.py

Plot the standalone spike-time and connectivity artifacts written by
rerun_trial_save_spikes_conn.py. Pure numpy/matplotlib — no NEURON/MPI, so it
runs on a login node:

    shifter --image=adammwea/netsims_docker:v1 python3 plot_spikes_conn.py \
        --output-dir op_300s_trial135 --label trial_135_rerun_300s

Produces:
    <label>_raster.png            spike raster (E blue / I red, sorted)
    <label>_popactivity.png       population spike-rate over time (E, I, total)
    <label>_connectivity.png      weighted connectivity matrix + degree/weight stats
    <label>_experimental_style.svg  raster + network-activity panels in the same
                                  layout as the experimental recordings, so the
                                  simulated figure can be compared side by side
                                  with plot_experimental_only output

The experimental-style panel reuses the exact helpers from fitnessFunc_v2 that
plot_experimental_only uses, so the two figures match. Those helpers pull in
NEURON via fitnessFunc_v2, so they are imported lazily — the three .png plots
above still work with nothing but numpy/matplotlib.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import numpy as np


def load(out_dir: Path, label: str):
    sp = np.load(out_dir / f'{label}_spikes.npz', allow_pickle=True)
    cn = np.load(out_dir / f'{label}_connectivity.npz', allow_pickle=True)
    return sp, cn


def plot_raster(sp, out_path: Path, duration_s=None, start_s=0.0):
    spkt = np.asarray(sp['spkt_s'], dtype=float)
    spkid = np.asarray(sp['spkid'], dtype=int)
    excit = set(int(g) for g in sp['excit_gids'])
    inhib = set(int(g) for g in sp['inhib_gids'])
    if duration_s is None:
        duration_s = float(spkt.max()) if len(spkt) else 1.0

    # Crop: exclude spikes before start_s (and after duration_s) from the plot.
    in_win = (spkt >= start_s) & (spkt <= duration_s)

    # y-order: E gids first (sorted), then I gids (sorted).
    e_sorted = sorted(excit)
    i_sorted = sorted(inhib)
    gid_to_y = {}
    for y, g in enumerate(e_sorted + i_sorted):
        gid_to_y[g] = y
    n_e = len(e_sorted)

    fig, ax = plt.subplots(figsize=(16, 8))
    is_inhib = np.array([g in inhib for g in spkid])
    y = np.array([gid_to_y.get(int(g), -1) for g in spkid])
    valid = (y >= 0) & in_win
    ax.plot(spkt[valid & ~is_inhib], y[valid & ~is_inhib], linestyle='None',
            marker='|', markersize=2, markeredgewidth=0.4, color='tab:blue', alpha=0.6, rasterized=True)
    ax.plot(spkt[valid & is_inhib], y[valid & is_inhib], linestyle='None',
            marker='|', markersize=2, markeredgewidth=0.4, color='tab:red', alpha=0.6, rasterized=True)
    ax.axhline(n_e - 0.5, color='k', lw=0.6, ls='--', alpha=0.5)

    ax.set_xlim(start_s, duration_s)
    ax.set_ylim(-1, len(gid_to_y))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Unit (E below dashed line, I above)')
    win_note = f' (cropped first {start_s:g} s)' if start_s > 0 else ''
    ax.set_title(f'Spike raster — {int(valid.sum())} spikes, {start_s:g}–{duration_s:g} s{win_note}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(handles=[
        Line2D([0], [0], marker='|', color='tab:blue', linestyle='None', markersize=9,
               markeredgewidth=1.5, label=f'Excitatory ({n_e})'),
        Line2D([0], [0], marker='|', color='tab:red', linestyle='None', markersize=9,
               markeredgewidth=1.5, label=f'Inhibitory ({len(i_sorted)})'),
    ], loc='upper right', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  saved {out_path}')


def plot_pop_activity(sp, out_path: Path, duration_s=None, bin_s=0.05, start_s=0.0):
    spkt = np.asarray(sp['spkt_s'], dtype=float)
    spkid = np.asarray(sp['spkid'], dtype=int)
    ctype = np.asarray(sp['cell_type'])
    n_e = len(sp['excit_gids'])
    n_i = len(sp['inhib_gids'])
    if duration_s is None:
        duration_s = float(spkt.max()) if len(spkt) else 1.0

    # Bins start at start_s, so spikes before the crop point fall outside every
    # bin and are excluded from all rate traces.
    bins = np.arange(start_s, duration_s + bin_s, bin_s)
    centers = 0.5 * (bins[:-1] + bins[1:])

    def rate(mask, npop):
        counts, _ = np.histogram(spkt[mask], bins=bins)
        return counts / bin_s / max(npop, 1)  # Hz per cell

    r_e = rate(ctype == 'E', n_e)
    r_i = rate(ctype == 'I', n_i)
    r_all = rate(np.ones_like(spkt, dtype=bool), n_e + n_i)

    # light smoothing (250 ms)
    win = max(1, int(0.25 / bin_s))
    k = np.ones(win) / win
    sm = lambda x: np.convolve(x, k, mode='same')

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(centers, sm(r_all), color='k', lw=1.4, label='All')
    ax.plot(centers, sm(r_e), color='tab:blue', lw=1.1, alpha=0.8, label='Excitatory')
    ax.plot(centers, sm(r_i), color='tab:red', lw=1.1, alpha=0.8, label='Inhibitory')
    ax.set_xlim(start_s, duration_s)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mean firing rate (Hz/cell)')
    win_note = f', cropped first {start_s:g} s' if start_s > 0 else ''
    ax.set_title(f'Population activity ({bin_s*1e3:g} ms bins, 250 ms smoothed{win_note})')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  saved {out_path}')


def plot_connectivity(cn, sp, out_path: Path):
    pre = np.asarray(cn['pre_gid'], dtype=int)
    post = np.asarray(cn['post_gid'], dtype=int)
    weight = np.asarray(cn['weight'], dtype=float)
    label = np.asarray(cn['label'])
    excit = set(int(g) for g in sp['excit_gids'])
    inhib = set(int(g) for g in sp['inhib_gids'])

    # Stable gid->index ordering: E first then I, so the matrix shows blocks.
    e_sorted = sorted(excit)
    i_sorted = sorted(inhib)
    order = e_sorted + i_sorted
    gid_to_idx = {g: i for i, g in enumerate(order)}
    n = len(order)
    n_e = len(e_sorted)

    # Summed synaptic weight per (pre, post) cell pair.
    W = np.zeros((n, n), dtype=float)
    for p, q, w in zip(pre, post, weight):
        ip = gid_to_idx.get(int(p))
        iq = gid_to_idx.get(int(q))
        if ip is not None and iq is not None:
            W[ip, iq] += w

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.4, 1, 1], height_ratios=[1, 1])

    # --- connectivity matrix ---
    ax0 = fig.add_subplot(gs[:, 0])
    Wpos = np.ma.masked_where(W <= 0, W)
    im = ax0.imshow(Wpos, aspect='auto', cmap='viridis',
                    norm=LogNorm(vmin=max(Wpos.min(), 1e-3), vmax=Wpos.max()),
                    origin='upper', interpolation='nearest')
    ax0.axhline(n_e - 0.5, color='w', lw=0.8, ls='--', alpha=0.7)
    ax0.axvline(n_e - 0.5, color='w', lw=0.8, ls='--', alpha=0.7)
    ax0.set_xlabel('Post-synaptic cell (E | I)')
    ax0.set_ylabel('Pre-synaptic cell (E | I)')
    ax0.set_title(f'Summed synaptic weight matrix\n{len(pre)} contacts, {n} cells')
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04, label='Σ weight (log)')

    # --- weight distribution by label ---
    ax1 = fig.add_subplot(gs[0, 1])
    colors = {'E->E': 'tab:blue', 'E->I': 'tab:cyan', 'I->E': 'tab:red', 'I->I': 'tab:orange'}
    for lab in ['E->E', 'E->I', 'I->E', 'I->I']:
        m = label == lab
        if m.sum():
            ax1.hist(weight[m], bins=60, histtype='step', lw=1.4,
                     color=colors.get(lab, 'gray'), label=f'{lab} ({m.sum()})')
    ax1.set_xlabel('Synaptic weight')
    ax1.set_ylabel('Count')
    ax1.set_yscale('log')
    ax1.set_title('Weight distribution by type')
    ax1.legend(fontsize=8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- in/out degree distributions ---
    out_deg = np.bincount(np.array([gid_to_idx[int(g)] for g in pre if int(g) in gid_to_idx]), minlength=n)
    in_deg = np.bincount(np.array([gid_to_idx[int(g)] for g in post if int(g) in gid_to_idx]), minlength=n)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(in_deg, bins=40, color='tab:green', alpha=0.7)
    ax2.set_xlabel('In-degree (contacts received)')
    ax2.set_ylabel('# cells')
    ax2.set_title(f'In-degree (mean {in_deg.mean():.0f})')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(out_deg, bins=40, color='tab:purple', alpha=0.7)
    ax3.set_xlabel('Out-degree (contacts made)')
    ax3.set_ylabel('# cells')
    ax3.set_title(f'Out-degree (mean {out_deg.mean():.0f})')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # --- contacts-by-type bar ---
    ax4 = fig.add_subplot(gs[1, 2])
    labs, counts = np.unique(label, return_counts=True)
    ax4.bar(range(len(labs)), counts, color=[colors.get(l, 'gray') for l in labs])
    ax4.set_xticks(range(len(labs)))
    ax4.set_xticklabels(labs, rotation=30)
    ax4.set_ylabel('# synaptic contacts')
    ax4.set_title('Contacts by connection type')
    for i, c in enumerate(counts):
        ax4.text(i, c, f'{c}', ha='center', va='bottom', fontsize=8)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    fig.suptitle('Network connectivity', fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  saved {out_path}')


def _load_burst_helpers():
    """Import the burst detector + panel helpers that plot_experimental_only uses.

    Imported lazily and only for the experimental-style figure: fitnessFunc_v2
    pulls in NEURON, and the rest of this script deliberately needs nothing but
    numpy/matplotlib so it stays runnable on a login node.
    """
    import os
    import sys
    rbs_root = '/pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models'
    for p in (rbs_root,
              os.path.join(rbs_root, 'RBS_network_models'),
              '/pscratch/sd/k/ktub1999/networkSimulations/MEA_Analysis/IPNAnalysis'):
        if p not in sys.path:
            sys.path.insert(0, p)
    from fitnessFunc_v2 import (_run_burst_detector, _plot_raster,
                                _plot_network_signal, _overlay_bursts)
    return _run_burst_detector, _plot_raster, _plot_network_signal, _overlay_bursts


def _compute_sim_burst_data(spike_data, run_burst_detector):
    """Two-pass (main + pre) burst detection, matching compute_sim_burst_data in
    rerun_trial_from_cfg_experimental_style.py so both figures mark the same events."""
    main = run_burst_detector(spike_data, min_burstlet_participation=0.01,
                              base_threshold_static=50)
    pre = run_burst_detector(spike_data, min_burstlet_participation=0.05,
                             base_threshold_static=15)
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


def plot_experimental_style(sp, out_path: Path, duration_s=None, start_s=0.0,
                            title_label=''):
    """Raster over network-activity, laid out like the experimental recordings.

    Mirrors plot_simulated_experimental_style() in
    rerun_trial_from_cfg_experimental_style.py, but reads the standalone spikes
    .npz instead of re-running the simulation.
    """
    (run_burst_detector, plot_raster_panel,
     plot_network_signal, overlay_bursts) = _load_burst_helpers()

    spkt = np.asarray(sp['spkt_s'], dtype=float)
    spkid = np.asarray(sp['spkid'], dtype=int)
    inhib = set(int(g) for g in sp['inhib_gids'])
    if duration_s is None:
        duration_s = float(spkt.max()) if len(spkt) else 1.0

    # Window, then rebase to t=0. The detector indexes its internal signal from
    # the start of the recording, so a window that begins at start_s > 0 walks
    # merge() off the end of that array and raises on a zero-size slice.
    in_win = (spkt >= start_s) & (spkt <= duration_s)
    t_win = spkt[in_win] - start_s
    id_win = spkid[in_win]
    span_s = float(duration_s - start_s)

    spike_data, cell_types = {}, {}
    for g in np.unique(id_win):
        gid = int(g)
        spike_data[gid] = t_win[id_win == g]
        cell_types[gid] = 'inhibitory' if gid in inhib else 'excitatory'

    burst = _compute_sim_burst_data(spike_data, run_burst_detector)

    fig, (ax_raster, ax_net) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    plot_raster_panel(ax_raster, spike_data,
                      title=f'Simulated Raster — {title_label}',
                      cell_types=cell_types)
    overlay_bursts(ax_raster, burst['burst_events'])

    if burst['plot_data']:
        plot_network_signal(ax_net, burst['plot_data'], title='Simulated Network Activity')
        overlay_bursts(ax_net, burst['burst_events'])
    else:
        ax_net.text(0.5, 0.5, 'No burst data', transform=ax_net.transAxes, ha='center')

    ax_net.set_xlabel('Time (s)')

    # --- tick / legend overrides copied from plot_experimental_only ---
    ax_raster.set_yticks([0, 200, 400])
    ax_raster.set_xticks([0, span_s])
    _, y_top = ax_raster.get_ylim()
    ax_raster.set_ylim(0, y_top)
    ax_raster.set_xlim(0, span_s)

    # Cap network-activity y-axis at 100
    ax_net.set_yticks([10, 50])
    ax_net.set_xticks([0, span_s])
    ax_net.set_ylim(0, 100)
    ax_net.set_xlim(0, span_s)

    for ax in (ax_raster, ax_net):
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.08)
    fig.savefig(out_path)
    plt.close(fig)
    n_bl = len(burst['burst_events'].get('burstlets', []))
    n_nb = len(burst['burst_events'].get('network_bursts', []))
    n_sb = len(burst['burst_events'].get('superbursts', []))
    print(f'  bursts: {n_bl} burstlets, {n_nb} network bursts, {n_sb} superbursts')
    print(f'  saved {out_path}')


def main():
    ap = argparse.ArgumentParser(description='Plot spike + connectivity npz artifacts.')
    ap.add_argument('--output-dir', required=True, help='Dir containing the *_spikes.npz / *_connectivity.npz')
    ap.add_argument('--label', required=True, help='File stem, e.g. trial_135_rerun_300s')
    ap.add_argument('--duration-seconds', type=float, default=None, help='X-axis limit; inferred from spikes if omitted.')
    ap.add_argument('--crop-start-seconds', type=float, default=0.0,
                    help='Exclude spikes before this time (s) from the raster and population-activity '
                         'plots — used to drop the startup transient. Connectivity is structural and '
                         'is unaffected. Default 0 (no crop).')
    ap.add_argument('--no-experimental-style', action='store_true',
                    help='Skip the experimental-style SVG (the only output that needs '
                         'fitnessFunc_v2 / MEA_Analysis on sys.path).')
    args = ap.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    label = args.label
    start_s = float(args.crop_start_seconds)
    sp, cn = load(out_dir, label)
    print(f'Loaded {len(sp["spkt_s"])} spikes, {len(cn["pre_gid"])} contacts from {out_dir}')
    if start_s > 0:
        print(f'Cropping first {start_s:g} s from raster + population activity '
              '(connectivity is structural, unaffected).')

    plot_raster(sp, out_dir / f'{label}_raster.png', args.duration_seconds, start_s)
    plot_pop_activity(sp, out_dir / f'{label}_popactivity.png', args.duration_seconds, start_s=start_s)
    plot_connectivity(cn, sp, out_dir / f'{label}_connectivity.png')

    if not args.no_experimental_style:
        try:
            plot_experimental_style(
                sp, out_dir / f'{label}_experimental_style.svg',
                args.duration_seconds, start_s, title_label=label)
        except ImportError as exc:
            print(f'  skipped experimental-style plot: {exc}\n'
                  '  (needs fitnessFunc_v2 + MEA_Analysis/IPNAnalysis on sys.path)')
    print('Done.')


if __name__ == '__main__':
    main()
