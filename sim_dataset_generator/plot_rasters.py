#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_rasters.py - Plot raster plots from HDF5 datasets produced by DatasetWriter.

Usage
-----
  # Plot first 5 simulations from an HDF5 file:
  python plot_rasters.py dataset.h5

  # Plot specific sim IDs:
  python plot_rasters.py /pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/cnn_dataset_49738154.h5

  # Plot all simulations (one figure per sim):
  python plot_rasters.py dataset.h5 --all

  # Use the binary raster matrix instead of spkt/spkid:
  python plot_rasters.py dataset.h5 --mode matrix

  # Save figures instead of showing:
  python plot_rasters.py dataset.h5 --save_dir ./plots
"""

import argparse
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_raster_spikes(ax, spkt, spkid, duration_ms,
                       n_excit=0, n_inhib=0, title=""):
    """Plot a raster from raw spike times and spike IDs.

    Excitatory cells (gid < n_excit) are drawn in blue;
    inhibitory cells (gid >= n_excit) are drawn in red.
    """
    if len(spkt) == 0:
        ax.text(
            0.5, 0.5, "No spikes", transform=ax.transAxes,
            ha="center", va="center", fontsize=12, color="gray",
        )
        ax.set_title(title, fontsize=10)
        return

    n_total = n_excit + n_inhib
    if n_excit > 0:
        exc_mask = spkid < n_excit
    else:
        exc_mask = np.ones(len(spkid), dtype=bool)
    inh_mask = ~exc_mask

    if exc_mask.any():
        ax.plot(
            spkt[exc_mask], spkid[exc_mask],
            "|", color="tab:blue", markersize=2, markeredgewidth=0.4,
            alpha=0.7, rasterized=True,
        )
    if inh_mask.any():
        ax.plot(
            spkt[inh_mask], spkid[inh_mask],
            "|", color="tab:red", markersize=2, markeredgewidth=0.4,
            alpha=0.7, rasterized=True,
        )

    ax.set_xlim(0, duration_ms)
    ax.set_ylim(-0.5, max(n_total, spkid.max() + 1) - 0.5)
    ax.set_ylabel("Cell ID")
    ax.set_title(title, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    handles = []
    if n_excit > 0:
        handles.append(
            Line2D([], [], marker="|", color="tab:blue", linestyle="None",
                   markersize=6, markeredgewidth=1.5,
                   label="Exc (%d)" % n_excit)
        )
    if n_inhib > 0:
        handles.append(
            Line2D([], [], marker="|", color="tab:red", linestyle="None",
                   markersize=6, markeredgewidth=1.5,
                   label="Inh (%d)" % n_inhib)
        )
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.7)


def plot_raster_matrix(ax, raster, duration_ms, bin_ms,
                       n_excit=0, n_inhib=0, title=""):
    """Plot a raster from the binary raster matrix [n_cells, n_timebins]."""
    if raster.size == 0:
        ax.text(
            0.5, 0.5, "No raster data", transform=ax.transAxes,
            ha="center", va="center", fontsize=12, color="gray",
        )
        ax.set_title(title, fontsize=10)
        return

    n_cells, n_bins = raster.shape

    # Build an RGB image so we can colour E/I differently
    img = np.zeros((n_cells, n_bins, 3), dtype=float)
    for c in range(n_cells):
        active = raster[c] > 0
        if n_excit > 0 and c < n_excit:
            img[c, active] = [0.12, 0.47, 0.71]  # tab:blue
        else:
            img[c, active] = [0.84, 0.15, 0.16]  # tab:red

    ax.imshow(
        img,
        aspect="auto",
        interpolation="none",
        origin="lower",
        extent=[0, duration_ms, -0.5, n_cells - 0.5],
    )
    ax.set_ylabel("Cell ID")
    ax.set_title(title, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_population_rate(ax, spkt, duration_ms, bin_ms=10.0,
                         n_cells=1, title=""):
    """Plot the population firing rate histogram below a raster."""
    if len(spkt) == 0:
        ax.set_title(title, fontsize=9)
        return

    bins = np.arange(0, duration_ms + bin_ms, bin_ms)
    counts, edges = np.histogram(spkt, bins=bins)
    rate = counts / (bin_ms / 1000.0) / max(n_cells, 1)  # Hz
    centres = (edges[:-1] + edges[1:]) / 2.0

    ax.fill_between(centres, rate, color="tab:gray", alpha=0.5)
    ax.plot(centres, rate, color="tab:gray", lw=0.8)
    ax.set_ylabel("Rate (Hz)")
    ax.set_xlim(0, duration_ms)
    ax.set_title(title, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Main plotting driver
# ---------------------------------------------------------------------------


def list_simulations(h5_path):
    """Return sorted list of simulation group keys in the HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        if "simulations" not in f:
            return []
        return sorted(f["simulations"].keys())


def plot_single_sim(h5_path, sim_key, mode="spikes", save_path=None):
    """Plot a single simulation from the HDF5 file.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 dataset file.
    sim_key : str
        Key like 'sim_00000'.
    mode : str
        'spikes' -- use spkt/spkid;  'matrix' -- use the raster dataset.
    save_path : str or None
        If given, save figure to this path; otherwise call plt.show().
    """
    with h5py.File(h5_path, "r") as f:
        grp = f["simulations/" + sim_key]
        spkt = grp["spkt"][:]
        spkid = grp["spkid"][:]
        raster = grp["raster"][:] if "raster" in grp else np.empty((0, 0))
        duration_ms = float(grp.attrs.get("duration_ms", 1000.0))
        n_excit = int(grp.attrs.get("n_excit", 0))
        n_inhib = int(grp.attrs.get("n_inhib", 0))
        status = grp.attrs.get("status", "unknown")
        sim_id = int(grp.attrs.get("sim_id", 0))

        # Read bin_ms from metadata if available
        bin_ms = 5.0
        if "metadata" in f:
            bin_ms = float(f["metadata"].attrs.get("bin_ms", 5.0))

    n_total = n_excit + n_inhib
    n_spikes = len(spkt)
    title_base = (
        "Sim %d (%s)  |  status=%s  |  %d spikes  |  %dE + %dI  |  %.0f ms"
        % (sim_id, sim_key, status, n_spikes, n_excit, n_inhib, duration_ms)
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax_raster, ax_rate = axes

    if mode == "matrix" and raster.size > 0:
        plot_raster_matrix(
            ax_raster, raster, duration_ms, bin_ms,
            n_excit=n_excit, n_inhib=n_inhib,
            title=title_base,
        )
    else:
        plot_raster_spikes(
            ax_raster, spkt, spkid, duration_ms,
            n_excit=n_excit, n_inhib=n_inhib,
            title=title_base,
        )

    plot_population_rate(
        ax_rate, spkt, duration_ms,
        bin_ms=10.0, n_cells=max(n_total, 1),
        title="Population Firing Rate",
    )
    ax_rate.set_xlabel("Time (ms)")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print("  Saved: %s" % save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_overview(h5_path, sim_keys, mode="spikes", save_path=None, cols=2):
    """Plot a grid overview of multiple simulations.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 dataset file.
    sim_keys : list of str
        Keys like ['sim_00000', 'sim_00001', ...].
    mode : str
        'spikes' or 'matrix'.
    save_path : str or None
        Save figure to this path; otherwise plt.show().
    cols : int
        Number of columns in the grid.
    """
    n = len(sim_keys)
    rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 3 * rows),
                             squeeze=False, sharex=False)

    with h5py.File(h5_path, "r") as f:
        bin_ms = 5.0
        if "metadata" in f:
            bin_ms = float(f["metadata"].attrs.get("bin_ms", 5.0))

        for idx, sim_key in enumerate(sim_keys):
            r, c = divmod(idx, cols)
            ax = axes[r][c]

            grp_path = "simulations/" + sim_key
            if grp_path not in f:
                ax.text(0.5, 0.5, "%s\nNOT FOUND" % sim_key,
                        transform=ax.transAxes, ha="center", va="center")
                continue

            grp = f[grp_path]
            spkt = grp["spkt"][:]
            spkid = grp["spkid"][:]
            raster = grp["raster"][:] if "raster" in grp else np.empty((0, 0))
            duration_ms = float(grp.attrs.get("duration_ms", 1000.0))
            n_excit = int(grp.attrs.get("n_excit", 0))
            n_inhib = int(grp.attrs.get("n_inhib", 0))
            status = grp.attrs.get("status", "unknown")
            sim_id = int(grp.attrs.get("sim_id", 0))
            n_spikes = len(spkt)

            title = (
                "Sim %d  |  %s  |  %d spk  |  %dE+%dI"
                % (sim_id, status, n_spikes, n_excit, n_inhib)
            )

            if mode == "matrix" and raster.size > 0:
                plot_raster_matrix(
                    ax, raster, duration_ms, bin_ms,
                    n_excit=n_excit, n_inhib=n_inhib, title=title,
                )
            else:
                plot_raster_spikes(
                    ax, spkt, spkid, duration_ms,
                    n_excit=n_excit, n_inhib=n_inhib, title=title,
                )
            ax.set_xlabel("Time (ms)")

    # Hide unused subplots
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle(
        "Raster Overview -- %s  (%d sims)" % (os.path.basename(h5_path), n),
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print("Saved overview: %s" % save_path)
        plt.close(fig)
    else:
        plt.show()


def print_h5_summary(h5_path):
    """Print a summary of the HDF5 dataset."""
    with h5py.File(h5_path, "r") as f:
        print("\n" + "=" * 60)
        print("HDF5 Dataset: %s" % h5_path)
        print("=" * 60)

        if "metadata" in f:
            meta = f["metadata"]
            print("  n_samples:   %s" % meta.attrs.get("n_samples", "?"))
            print("  bin_ms:      %s" % meta.attrs.get("bin_ms", "?"))
            print("  date:        %s" % meta.attrs.get("date", "?"))
            print("  completed:   %s" % meta.attrs.get("completed", False))
            pnames = meta.attrs.get("param_names", [])
            if len(pnames) > 0:
                shown = list(pnames[:5])
                suffix = "..." if len(pnames) > 5 else ""
                print("  params (%d): %s%s" % (len(pnames), shown, suffix))

        if "simulations" in f:
            sim_keys = sorted(f["simulations"].keys())
            n_sims = len(sim_keys)
            n_ok = sum(
                1 for k in sim_keys
                if f["simulations/" + k].attrs.get("status") == "ok"
            )
            n_failed = n_sims - n_ok
            print("  simulations: %d  (%d ok, %d failed)" % (n_sims, n_ok, n_failed))

            # Spike count summary for first few
            spike_counts = []
            for k in sim_keys[:20]:
                spkt = f["simulations/" + k + "/spkt"]
                spike_counts.append(len(spkt))
            if spike_counts:
                print("  spike counts (first %d): min=%d, max=%d, mean=%.0f"
                      % (len(spike_counts), min(spike_counts),
                         max(spike_counts), np.mean(spike_counts)))
        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot raster plots from HDF5 datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("h5_path", type=str, help="Path to the HDF5 dataset file.")
    parser.add_argument(
        "--sims", type=int, nargs="+", default=None,
        help="Specific sim IDs to plot (e.g. --sims 0 3 7). Default: first 5.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Plot all simulations (individual figures).",
    )
    parser.add_argument(
        "--overview", action="store_true", default=True,
        help="Plot a grid overview of selected sims (default).",
    )
    parser.add_argument(
        "--individual", action="store_true",
        help="Also save individual per-sim figures.",
    )
    parser.add_argument(
        "--mode", type=str, choices=["spikes", "matrix"], default="spikes",
        help="'spikes' uses spkt/spkid; 'matrix' uses the raster array.",
    )
    parser.add_argument(
        "--save_dir", type=str, default=None,
        help="Directory to save plots. If not set, saves next to the HDF5 file.",
    )
    parser.add_argument(
        "--max_sims", type=int, default=15,
        help="Max number of sims to plot when --sims is not specified (default: 5).",
    )
    parser.add_argument(
        "--cols", type=int, default=2,
        help="Number of columns in the overview grid (default: 2).",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print dataset summary info.",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.h5_path):
        print("ERROR: File not found: %s" % args.h5_path)
        return

    if args.summary:
        print_h5_summary(args.h5_path)

    # Determine which sims to plot
    all_keys = list_simulations(args.h5_path)
    if not all_keys:
        print("No simulations found in the HDF5 file.")
        return

    if not args.summary:
        print_h5_summary(args.h5_path)

    if args.sims is not None:
        sim_keys = ["sim_%05d" % s for s in args.sims]
    elif args.all:
        sim_keys = all_keys
    else:
        sim_keys = all_keys[:args.max_sims]

    # Default save_dir: same directory as the HDF5 file
    if args.save_dir is None:
        args.save_dir = os.path.join(
            os.path.dirname(os.path.abspath(args.h5_path)), "raster_plots"
        )
        print("No --save_dir specified, saving to: %s" % args.save_dir)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    base = os.path.splitext(os.path.basename(args.h5_path))[0]

    print("Plotting %d simulation(s)  [mode=%s]" % (len(sim_keys), args.mode))

    # Overview grid
    if args.overview and len(sim_keys) > 1:
        save_path = os.path.join(args.save_dir, base + "_overview.png")
        plot_overview(
            args.h5_path, sim_keys, mode=args.mode,
            save_path=save_path, cols=args.cols,
        )

    # Individual figures
    if args.individual or len(sim_keys) == 1:
        for sim_key in sim_keys:
            save_path = os.path.join(args.save_dir, sim_key + "_raster.png")
            plot_single_sim(
                args.h5_path, sim_key, mode=args.mode, save_path=save_path,
            )


if __name__ == "__main__":
    main()
