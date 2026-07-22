"""
NetPyNE simulation runner for CNN-training dataset generation.

This module programmatically drives NetPyNE network simulations by:
1. Loading a model's cfg.py / netParams.py from a specified source directory,
2. Injecting evolutionary-parameter overrides into the SimConfig object,
3. Running the simulation via ``sim.createSimulateAnalyze``, and
4. Extracting spike rasters (spkt / spkid) suitable for downstream CNN training.

The runner is designed to be called repeatedly within a single MPI rank.
Each invocation reloads the cfg / netParams modules so that per-call
randomisation side-effects in cfg.py are properly re-executed.

Typical usage (inside an MPI worker loop)::

    result = run_single_sim(
        cfg_overrides={"weight_EE": 0.0035, "prob_EI": 0.12, ...},
        sim_id=42,
        model_src_dir="/path/to/models/.../src",
        duration_ms=25000.0,
    )
    if result.status == "ok":
        raster = rasterize_spikes(
            result.spkt, result.spkid,
            n_cells=result.n_excit + result.n_inhib,
            duration_ms=result.duration_ms,
        )
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
import math
import sys
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SimResult:
    """Container for a single simulation's inputs and outputs."""

    sim_id: int
    unit_params: np.ndarray        # shape [P], float32 — params in [0,1]
    physical_params: np.ndarray    # shape [P], float32 — params in physical units
    spkt: np.ndarray               # spike times (ms), float32, variable length
    spkid: np.ndarray              # cell GIDs, int32, variable length
    n_excit: int                   # number of excitatory cells
    n_inhib: int                   # number of inhibitory cells
    duration_ms: float             # simulation duration in ms
    status: str                    # 'ok' or 'failed'
    error_msg: str                 # empty string if ok, error message if failed


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------

def run_single_sim(
    cfg_overrides: Dict[str, float],
    sim_id: int,
    model_src_dir: str,
    duration_ms: float = 25000.0,
    unit_params: Optional[np.ndarray] = None,
) -> SimResult:
    """Run one NetPyNE simulation with the given parameter overrides.

    Parameters
    ----------
    cfg_overrides:
        Mapping of cfg attribute names to their *physical* values.  Each
        key-value pair is applied via ``setattr(cfg, key, val)`` before the
        simulation is launched.
    sim_id:
        An integer identifier for this simulation (used for bookkeeping).
    model_src_dir:
        Absolute path to the directory containing ``cfg.py`` and
        ``netParams.py`` (the model source).
    duration_ms:
        Simulation duration in milliseconds.  Overrides ``cfg.duration``.
    unit_params:
        Optional unit-scaled parameter vector (values in [0, 1]).  Stored
        in the returned :class:`SimResult` for traceability.  If *None*,
        an empty array is stored.

    Returns
    -------
    SimResult
        Populated result dataclass.  ``status`` is ``'ok'`` on success,
        ``'failed'`` otherwise.
    """

    # Lazy import — avoids import errors when the module is loaded outside
    # a NEURON-capable environment (e.g. during unit tests).
    from netpyne import sim  # noqa: F811

    if unit_params is None:
        unit_params = np.array([], dtype=np.float32)

    physical_params = np.array(
        list(cfg_overrides.values()), dtype=np.float32
    )

    # ------------------------------------------------------------------
    # 1. Ensure model_src_dir is on sys.path
    # ------------------------------------------------------------------
    if model_src_dir not in sys.path:
        sys.path.insert(0, model_src_dir)

    try:
        # --------------------------------------------------------------
        # 2. Clear NetPyNE state from any previous simulation
        # --------------------------------------------------------------
        try:
            sim.clearAll()
        except Exception:
            # clearAll may fail if no sim has been created yet; safe to ignore.
            pass

        # --------------------------------------------------------------
        # 3. (Re-)load cfg and netParams modules
        # --------------------------------------------------------------
        # First load evol_params to inject params into __main__
        try:
            params_mod = _reload_module("evol_params", model_src_dir)
            import __main__ as _main
            _main.params = params_mod.params
        except Exception:
            # evol_params might not exist or params might be defined elsewhere
            pass

        cfg_mod = _reload_module("cfg", model_src_dir)
        cfg = cfg_mod.cfg

        
        # --------------------------------------------------------------
        # 4. Apply parameter overrides (Before loading netParams)
        # --------------------------------------------------------------
        for key, val in cfg_overrides.items():
            setattr(cfg, key, val)
        # for key,val in params_mod.params.items():
        #     setattr(cfg, key, val)

        # Override duration
        cfg.duration = duration_ms

        # --- PERFORMANCE OPTIMIZATIONS ---
        # Disable saving, rendering, or recording traces to heavily save time
        cfg.saveData = False
        cfg.saveMat = False
        cfg.saveHDF5 = False
        cfg.saveDpk = False
        cfg.saveJson = False
        cfg.savePickle = False
        cfg.analysis = {}
        cfg.recordTraces = {}

        # Inject cfg into __main__ so that netParams.py's
        # "from __main__ import cfg" succeeds (standard NetPyNE pattern).
        import __main__ as _main
        _main.cfg = cfg

        netParams_mod = _reload_module("netParams", model_src_dir)
        netParams = netParams_mod.netParams

        # --------------------------------------------------------------
        # 5. Run simulation
        # --------------------------------------------------------------
        logger.info("sim_id=%d  starting simulation (duration=%.1f ms)", sim_id, duration_ms)
        sim.createSimulateAnalyze(simConfig=cfg, netParams=netParams)

        # --------------------------------------------------------------
        # 6. Extract results
        # --------------------------------------------------------------
        spkt = np.array(sim.simData["spkt"], dtype=np.float32)
        spkid = np.array(sim.simData["spkid"], dtype=np.int32)

        n_excit = len(getattr(cfg, "excit_units", []))
        n_inhib = len(getattr(cfg, "inhib_units", []))

        logger.info(
            "sim_id=%d  finished — %d spikes, %d excit cells, %d inhib cells",
            sim_id, len(spkt), n_excit, n_inhib,
        )

        return SimResult(
            sim_id=sim_id,
            unit_params=unit_params,
            physical_params=physical_params,
            spkt=spkt,
            spkid=spkid,
            n_excit=n_excit,
            n_inhib=n_inhib,
            duration_ms=duration_ms,
            status="ok",
            error_msg="",
        )

    except Exception as exc:
        logger.error("sim_id=%d  FAILED: %s", sim_id, exc, exc_info=True)
        return SimResult(
            sim_id=sim_id,
            unit_params=unit_params,
            physical_params=physical_params,
            spkt=np.array([], dtype=np.float32),
            spkid=np.array([], dtype=np.int32),
            n_excit=0,
            n_inhib=0,
            duration_ms=duration_ms,
            status="failed",
            error_msg=str(exc),
        )


# ---------------------------------------------------------------------------
# Spike rasterisation
# ---------------------------------------------------------------------------

def rasterize_spikes(
    spkt: np.ndarray,
    spkid: np.ndarray,
    n_cells: int,
    duration_ms: float,
    bin_ms: float = 5.0,
) -> np.ndarray:
    """Convert raw spike trains into a 2-D raster image.

    Parameters
    ----------
    spkt:
        1-D array of spike times in milliseconds.
    spkid:
        1-D array of cell GIDs corresponding to *spkt*.
    n_cells:
        Total number of cells in the network (rows of the output).
    duration_ms:
        Total simulation duration in milliseconds.
    bin_ms:
        Width of each time bin in milliseconds (default 5 ms).

    Returns
    -------
    np.ndarray
        ``uint8`` array of shape ``[n_cells, n_timebins]`` where
        ``n_timebins = ceil(duration_ms / bin_ms)``.  Each entry is the
        spike count for that cell / time-bin, clipped to 255.
    """

    n_timebins = math.ceil(duration_ms / bin_ms)
    raster = np.zeros((n_cells, n_timebins), dtype=np.uint16)

    if len(spkt) == 0:
        return raster.astype(np.uint8)

    spkt = np.asarray(spkt, dtype=np.float64)
    spkid = np.asarray(spkid, dtype=np.int64)

    # Compute time-bin indices (0-based)
    bin_idx = np.floor(spkt / bin_ms).astype(np.int64)
    # Clip to valid range (spikes exactly at duration_ms edge)
    np.clip(bin_idx, 0, n_timebins - 1, out=bin_idx)

    # Clip cell ids to valid range
    valid = (spkid >= 0) & (spkid < n_cells)
    if not np.all(valid):
        logger.warning(
            "rasterize_spikes: %d spikes with out-of-range GIDs dropped",
            int((~valid).sum()),
        )
        spkid = spkid[valid]
        bin_idx = bin_idx[valid]

    # Vectorised accumulation
    np.add.at(raster, (spkid, bin_idx), 1)

    # Clip to uint8 range
    np.clip(raster, 0, 255, out=raster)
    return raster.astype(np.uint8)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _reload_module(module_name: str, src_dir: str):
    """Import (or reload) *module_name* from *src_dir*.

    On the first call the module is imported normally.  On subsequent calls
    ``importlib.reload`` is used so that module-level side effects (e.g.
    random parameter sampling in ``cfg.py``) are re-executed.

    If the module was previously imported from a *different* directory we
    first evict it from ``sys.modules`` to avoid stale references.
    """

    # Evict any previously-cached version that might come from a different
    # directory (or simply to guarantee a full re-execution).
    if module_name in sys.modules:
        existing = sys.modules[module_name]
        existing_file = getattr(existing, "__file__", "") or ""
        # Always remove — we want a clean re-execution every call.
        del sys.modules[module_name]
        logger.debug(
            "Evicted cached module '%s' (was %s)", module_name, existing_file
        )

    mod = importlib.import_module(module_name)
    return mod
