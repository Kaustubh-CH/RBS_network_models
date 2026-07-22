#!/usr/bin/env python3
"""
run_dataset.py — CLI entry point for the RBS network dataset generator.

Generates an HDF5 dataset of network simulation results by:
  1. Sampling parameters (uniform or Latin Hypercube) from the defined
     parameter space.
  2. Running NetPyNE simulations in parallel across MPI ranks.
  3. Rasterising spike output into binary images.
  4. Writing everything into a single merged HDF5 file.

Usage (single process):
    python run_dataset.py --n_samples 100 --output dataset.h5 \\
                          --model_dir /path/to/model/src

Usage (MPI on Perlmutter):
    srun -n 64 python run_dataset.py --n_samples 1000 --output dataset.h5 \\
                                     --model_dir /path/to/model/src --lhs
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

from sim_dataset_generator.param_space import (
    get_param_names,
    get_bounds,
    get_n_params,
)
from sim_dataset_generator.param_sampling import (
    sample_unit_params,
    unit_to_physical,
    physical_to_cfg_dict,
)
from sim_dataset_generator.sim_runner import (
    run_single_sim,
    rasterize_spikes,
)
from sim_dataset_generator.data_writer import (
    DatasetWriter,
    merge_rank_files,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate an HDF5 dataset of RBS network simulations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n_samples", type=int, required=True,
        help="Number of simulations to run.",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output HDF5 file.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--bin_ms", type=float, default=5.0,
        help="Time bin width (ms) for raster images.",
    )
    parser.add_argument(
        "--lhs", action="store_true", default=False,
        help="Use Latin Hypercube Sampling.",
    )
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Path to model src/ directory containing cfg.py and netParams.py.",
    )
    parser.add_argument(
        "--duration_ms", type=float, default=25000.0,
        help="Simulation duration in ms.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Overwrite existing output file.",
    )
    parser.add_argument(
        "--dry_run", action="store_true", default=False,
        help="Sample params and show stats without running sims.",
    )
    parser.add_argument(
        "--no_mpi", action="store_true", default=False,
        help="Disable MPI and append the hdf5 output with the SLURM process rank.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    wall_t0 = time.time()
    args = parse_args()

    # --- MPI setup ---
    if args.no_mpi:
        comm = None
        rank = int(os.environ.get('SLURM_PROCID', '0'))
        n_ranks = int(os.environ.get('SLURM_NTASKS', '1'))
        logging.info(f"Running in no-MPI mode with rank {rank} out of {n_ranks} (if SLURM env vars are set)")
    else:
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            n_ranks = comm.Get_size()
        except ImportError:
            rank = 0
            n_ranks = 1
            comm = None

    # --- Logging ---
    log_level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format=f"[rank {rank}] %(asctime)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Ensure output directory exists (rank 0 only) ---
    if rank == 0:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    # --- Step 1: Parameter sampling (all ranks, deterministic) ---
    param_names = get_param_names()
    bounds = get_bounds()
    n_params = get_n_params()

    # Modify seed slightly per rank in no_mpi mode so all 10 processes don't do the exact same work
    current_seed = args.seed
    if args.no_mpi and current_seed is not None and n_ranks > 1:
        current_seed += rank

    unit_params = sample_unit_params(
        args.n_samples, n_params, seed=current_seed, use_lhs=args.lhs,
    )
    physical_params = unit_to_physical(unit_params, bounds)

    # --- Dry run mode ---
    if args.dry_run:
        if rank == 0:
            print(f"Parameter space: {n_params} parameters")
            print(f"Samples: {args.n_samples}")
            print(f"Physical param ranges (sampled):")
            for i, name in enumerate(param_names):
                col = physical_params[:, i]
                print(
                    f"  {name:30s}  [{col.min():.4f}, {col.max():.4f}]  "
                    f"(bounds: [{bounds[i, 0]:.4f}, {bounds[i, 1]:.4f}])"
                )
            elapsed = time.time() - wall_t0
            print(f"\nDry-run completed in {elapsed:.2f} s")
        return

    # --- Step 2: Distribute work across MPI ranks ---
    all_indices = np.arange(args.n_samples)
    if args.no_mpi:
        rank_indices = all_indices
    else:
        rank_indices = np.array_split(all_indices, n_ranks)[rank]

    if rank == 0 or args.no_mpi:
        logging.info(f"Running {len(rank_indices)} simulations (Rank {rank}/{n_ranks})")
        logging.info(f"Output: {args.output}")

    # --- Step 3: Run simulations ---
    results = []
    rasters = []

    # Progress bar only on rank 0
    iterator = rank_indices
    if rank == 0:
        try:
            from tqdm import tqdm
            iterator = tqdm(rank_indices, desc="Simulations", unit="sim")
        except ImportError:
            pass

    for sim_id in iterator:
        sim_id = int(sim_id)
        cfg_overrides = physical_to_cfg_dict(physical_params[sim_id], param_names)

        result = run_single_sim(
            cfg_overrides=cfg_overrides,
            sim_id=sim_id,
            model_src_dir=args.model_dir,
            duration_ms=args.duration_ms,
        )
        # Attach param vectors to result
        result.unit_params = unit_params[sim_id]
        result.physical_params = physical_params[sim_id]

        # Generate raster
        if result.status == "ok":
            n_cells = result.n_excit + result.n_inhib
            raster = rasterize_spikes(
                result.spkt, result.spkid,
                n_cells, result.duration_ms, args.bin_ms,
            )
        else:
            raster = np.zeros((1, 1), dtype=np.uint8)

        results.append(result)
        rasters.append(raster)

    # --- Step 4: Write results ---
    if n_ranks > 1 and comm is not None:
        # Each rank writes its own temp file
        rank_output = args.output.replace(".h5", f"_rank{rank:04d}.h5")
        writer = DatasetWriter(
            output_path=rank_output,
            n_samples=len(rank_indices),
            n_params=n_params,
            param_names=param_names,
            bounds=bounds,
            bin_ms=args.bin_ms,
            overwrite=True,
        )
        writer.create_file()
        writer.write_params(
            unit_params[rank_indices],
            physical_params[rank_indices],
        )
        writer.write_batch(results, rasters)

        # Barrier: wait for all ranks to finish writing
        comm.Barrier()

        # Rank 0 merges
        if rank == 0:
            rank_files = [
                args.output.replace(".h5", f"_rank{r:04d}.h5")
                for r in range(n_ranks)
            ]
            merge_rank_files(
                rank_files=rank_files,
                output_path=args.output,
                n_samples=args.n_samples,
                n_params=n_params,
                param_names=param_names,
                bounds=bounds,
                bin_ms=args.bin_ms,
            )
            logging.info(f"Dataset written to {args.output}")
    else:
        # Single process or no_mpi: write directly
        out_file = args.output
        if args.no_mpi and n_ranks > 1:
            out_file = args.output.replace(".h5", f"_rank{rank:04d}.h5")
            
        writer = DatasetWriter(
            output_path=out_file,
            n_samples=len(rank_indices),
            n_params=n_params,
            param_names=param_names,
            bounds=bounds,
            bin_ms=args.bin_ms,
            overwrite=args.overwrite or (args.no_mpi and n_ranks > 1),
        )
        writer.create_file()
        writer.write_params(
            unit_params[rank_indices] if n_ranks > 1 else unit_params,
            physical_params[rank_indices] if n_ranks > 1 else physical_params
        )
        writer.write_batch(results, rasters)
        writer.finalize()
        logging.info(f"Dataset written to {out_file}")

    # --- Timing summary ---
    elapsed = time.time() - wall_t0
    if rank == 0:
        logging.info(f"Total wall-clock time: {elapsed:.1f} s  "
                      f"({elapsed / 60:.2f} min)")


if __name__ == "__main__":
    main()
