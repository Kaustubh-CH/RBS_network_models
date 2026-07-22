"""
data_writer.py – Write simulation results to HDF5 format for CNN training.

HDF5 Layout
------------
dataset.h5
├── /metadata
│     attrs: date, n_samples, param_names, bounds,
│            git_hash, netpyne_version, bin_ms
│
├── /params
│   ├── unit_params        float32  shape [N, P]
│   └── physical_params    float32  shape [N, P]
│
└── /simulations
    ├── /sim_00000
    │   ├── spkt           float32  variable-length
    │   ├── spkid          int32    variable-length
    │   ├── raster         uint8    shape [n_cells, n_timebins]
    │   └── attrs: sim_id, duration_ms, n_excit,
    │              n_inhib, status ('ok' or 'failed')
    ├── /sim_00001
    │   └── ...
    └── ...
"""

from __future__ import annotations

import dataclasses
import datetime
import logging
import os
import subprocess
from typing import TYPE_CHECKING, List

import h5py
import numpy as np

if TYPE_CHECKING:
    from .sim_runner import SimResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_hash() -> str:
    """Return the current git HEAD hash, or 'unknown' on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _netpyne_version() -> str:
    """Return the installed netpyne version, or 'unknown' on failure."""
    try:
        import netpyne  # type: ignore
        return str(netpyne.__version__)
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# DatasetWriter
# ---------------------------------------------------------------------------


class DatasetWriter:
    """Writes simulation results to a single HDF5 file."""

    def __init__(
        self,
        output_path: str,
        n_samples: int,
        n_params: int,
        param_names: List[str],
        bounds: np.ndarray,
        bin_ms: float = 5.0,
        overwrite: bool = False,
    ) -> None:
        self.output_path = output_path
        self.n_samples = n_samples
        self.n_params = n_params
        self.param_names = param_names
        self.bounds = np.asarray(bounds, dtype=np.float32)
        self.bin_ms = bin_ms

        if os.path.exists(output_path):
            if overwrite:
                os.remove(output_path)
                logger.info("Removed existing file: %s", output_path)
            else:
                raise FileExistsError(
                    f"Output file already exists: {output_path}. "
                    "Set overwrite=True to replace it."
                )

    # ----- file creation ---------------------------------------------------

    def create_file(self) -> None:
        """Create the HDF5 file with metadata, param datasets, and
        an empty simulations group."""
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)

        with h5py.File(self.output_path, "w") as f:
            # -- /metadata --------------------------------------------------
            meta = f.create_group("metadata")
            meta.attrs["date"] = datetime.datetime.utcnow().isoformat()
            meta.attrs["n_samples"] = self.n_samples
            meta.attrs["n_params"] = self.n_params
            # Store param_names as an array of bytes (h5py compatible)
            meta.attrs["param_names"] = np.array(
                self.param_names, dtype=h5py.string_dtype()
            )
            meta.attrs["bounds"] = self.bounds  # [P, 2]
            meta.attrs["bin_ms"] = self.bin_ms
            meta.attrs["git_hash"] = _git_hash()
            meta.attrs["netpyne_version"] = _netpyne_version()

            # -- /params ----------------------------------------------------
            params_grp = f.create_group("params")
            params_grp.create_dataset(
                "unit_params",
                shape=(self.n_samples, self.n_params),
                dtype=np.float32,
                chunks=True,
                compression="gzip",
                compression_opts=4,
            )
            params_grp.create_dataset(
                "physical_params",
                shape=(self.n_samples, self.n_params),
                dtype=np.float32,
                chunks=True,
                compression="gzip",
                compression_opts=4,
            )

            # -- /simulations (empty) ---------------------------------------
            f.create_group("simulations")

        logger.info("Created HDF5 file: %s", self.output_path)

    # ----- parameter writing -----------------------------------------------

    def write_params(
        self, unit_params: np.ndarray, physical_params: np.ndarray
    ) -> None:
        """Write full parameter arrays to the existing datasets."""
        unit_params = np.asarray(unit_params, dtype=np.float32)
        physical_params = np.asarray(physical_params, dtype=np.float32)

        expected_shape = (self.n_samples, self.n_params)
        if unit_params.shape != expected_shape:
            raise ValueError(
                f"unit_params shape {unit_params.shape} != expected {expected_shape}"
            )
        if physical_params.shape != expected_shape:
            raise ValueError(
                f"physical_params shape {physical_params.shape} != expected "
                f"{expected_shape}"
            )

        with h5py.File(self.output_path, "r+") as f:
            f["params/unit_params"][...] = unit_params
            f["params/physical_params"][...] = physical_params

        logger.info("Wrote parameter arrays (%d samples).", self.n_samples)

    # ----- single simulation writing ---------------------------------------

    def write_simulation(
        self,
        sim_id: int,
        result: "SimResult",
        raster: np.ndarray,
    ) -> None:
        """Write one simulation result into the HDF5 file."""
        with h5py.File(self.output_path, "r+") as f:
            self._write_simulation_to_file(f, sim_id, result, raster)

    def _write_simulation_to_file(
        self,
        f: h5py.File,
        sim_id: int,
        result: "SimResult",
        raster: np.ndarray,
    ) -> None:
        """Internal helper – writes one sim group; *f* must be open in 'r+'."""
        grp_name = f"simulations/sim_{sim_id:05d}"
        try:
            if grp_name in f:
                logger.warning(
                    "Group %s already exists – overwriting.", grp_name
                )
                del f[grp_name]

            grp = f.create_group(grp_name)

            is_ok = result.status == "ok"

            # -- spike data -------------------------------------------------
            if is_ok and result.spkt is not None and len(result.spkt) > 0:
                spkt = np.asarray(result.spkt, dtype=np.float32)
                spkid = np.asarray(result.spkid, dtype=np.int32)
            else:
                spkt = np.empty(0, dtype=np.float32)
                spkid = np.empty(0, dtype=np.int32)

            grp.create_dataset("spkt", data=spkt)
            grp.create_dataset("spkid", data=spkid)

            # -- raster -----------------------------------------------------
            if is_ok and raster is not None and raster.size > 0:
                raster = np.asarray(raster, dtype=np.uint8)
            else:
                # For failed sims, store a minimal zeros raster
                raster = np.zeros((0, 0), dtype=np.uint8)

            grp.create_dataset(
                "raster",
                data=raster,
                compression="gzip",
                compression_opts=4,
            )

            # -- attributes -------------------------------------------------
            grp.attrs["sim_id"] = int(sim_id)
            grp.attrs["duration_ms"] = float(result.duration_ms)
            grp.attrs["n_excit"] = int(result.n_excit)
            grp.attrs["n_inhib"] = int(result.n_inhib)
            grp.attrs["status"] = result.status

        except Exception:
            logger.exception(
                "Error writing simulation %d – skipping.", sim_id
            )

    # ----- batch writing ---------------------------------------------------

    def write_batch(
        self,
        results: List["SimResult"],
        rasters: List[np.ndarray],
    ) -> None:
        """Write multiple simulation results in a single file open/close."""
        if len(results) != len(rasters):
            raise ValueError(
                f"results ({len(results)}) and rasters ({len(rasters)}) "
                "must have the same length."
            )
        with h5py.File(self.output_path, "r+") as f:
            for result, raster in zip(results, rasters):
                self._write_simulation_to_file(
                    f, result.sim_id, result, raster
                )
        logger.info("Wrote batch of %d simulations.", len(results))

    # ----- finalize --------------------------------------------------------

    def finalize(self) -> None:
        """Verify completeness and mark the dataset as finished."""
        with h5py.File(self.output_path, "r+") as f:
            sims_grp = f["simulations"]
            n_written = len(sims_grp)

            if n_written != self.n_samples:
                logger.warning(
                    "Expected %d simulations but found %d.",
                    self.n_samples,
                    n_written,
                )

            n_ok = sum(
                1
                for key in sims_grp
                if sims_grp[key].attrs.get("status") == "ok"
            )
            n_failed = n_written - n_ok

            f["metadata"].attrs["completed"] = True

        logger.info(
            "Finalized %s: %d/%d simulations written (%d ok, %d failed).",
            self.output_path,
            n_written,
            self.n_samples,
            n_ok,
            n_failed,
        )


# ---------------------------------------------------------------------------
# MPI merge helper
# ---------------------------------------------------------------------------


def merge_rank_files(
    rank_files: List[str],
    output_path: str,
    n_samples: int,
    n_params: int,
    param_names: List[str],
    bounds: np.ndarray,
    bin_ms: float,
) -> None:
    """Merge per-rank HDF5 files into a single dataset file.

    Each MPI rank writes its own temporary HDF5 file.  Rank 0 calls this
    function after all ranks are finished to combine them.

    Parameters
    ----------
    rank_files : list of str
        Paths to the per-rank HDF5 files, in rank order.
    output_path : str
        Path for the final merged HDF5 file.
    n_samples, n_params, param_names, bounds, bin_ms
        Metadata for the merged file (same as ``DatasetWriter.__init__``).
    """
    bounds = np.asarray(bounds, dtype=np.float32)

    writer = DatasetWriter(
        output_path=output_path,
        n_samples=n_samples,
        n_params=n_params,
        param_names=param_names,
        bounds=bounds,
        bin_ms=bin_ms,
        overwrite=True,
    )
    writer.create_file()

    # -- copy params from rank 0 -------------------------------------------
    with h5py.File(rank_files[0], "r") as src, \
         h5py.File(output_path, "r+") as dst:
        if "params/unit_params" in src:
            dst["params/unit_params"][...] = src["params/unit_params"][...]
            dst["params/physical_params"][...] = src["params/physical_params"][...]
            logger.info("Copied parameter arrays from %s.", rank_files[0])

    # -- copy simulation groups from every rank ----------------------------
    sim_count = 0
    with h5py.File(output_path, "r+") as dst:
        for rank_idx, rank_file in enumerate(rank_files):
            try:
                with h5py.File(rank_file, "r") as src:
                    if "simulations" not in src:
                        logger.warning(
                            "Rank file %s has no /simulations group – skipping.",
                            rank_file,
                        )
                        continue
                    for sim_key in sorted(src["simulations"].keys()):
                        src.copy(
                            f"simulations/{sim_key}",
                            dst["simulations"],
                            name=sim_key,
                        )
                        sim_count += 1
            except Exception:
                logger.exception(
                    "Error reading rank file %s – skipping.", rank_file
                )

    logger.info(
        "Merged %d simulations from %d rank files into %s.",
        sim_count,
        len(rank_files),
        output_path,
    )

    # -- finalize -----------------------------------------------------------
    writer.finalize()

    # -- remove temp files --------------------------------------------------
    for rank_file in rank_files:
        try:
            os.remove(rank_file)
            logger.debug("Deleted temp rank file: %s", rank_file)
        except OSError:
            logger.warning("Could not delete temp file: %s", rank_file)

    logger.info("Merge complete: %s", output_path)
