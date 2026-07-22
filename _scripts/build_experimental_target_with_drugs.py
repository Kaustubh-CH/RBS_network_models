#!/usr/bin/env python3
"""Build a baseline experimental target h5 AND stamp drug-response ratios in one shot.

This is a thin orchestrator over the two existing tools:

  1) ``create_experimental_target.py``  -- builds the per-unit baseline target h5
     (spike_times + features + network_results), with optional T_target truncation
     and location deduplication.
  2) ``extract_drug_effects.py``        -- for each drug, runs MUA threshold-detection
     on a (baseline_raw, drug_raw, well) tuple, computes the 12 post/pre rate-feature
     ratios, and writes them as ``/drug_effects/<drug_name>/`` into the baseline h5.

The point of bundling them is so the user doesn't have to (a) remember to run two
scripts in sequence and (b) hand-stamp drug ratios that came from a different well
(the "cross-well-stamping" case in the multi-drug optimization plan).

Each ``--drug NAME:BASELINE_RAW:DRUG_RAW:WELL`` flag is independent: drugs can
come from different wells / different recordings, but all land in the same
baseline target h5. Drug names are validated against ``DRUG_REGISTRY`` from the
active model's ``drug_perturbations.py`` BEFORE any work begins, so typos die
fast and we don't waste 20 minutes of MUA detection on a misspelled drug.

Example
-------

::

    python build_experimental_target_with_drugs.py \\
        --spike_times      .../well000/spike_times.npy \\
        --metrics          .../well000/metrics_curated.xlsx \\
        --network_results  .../well000/network_results.json \\
        --output           .../processed_experimental_targets/CDKL5_002_well000_unified.h5 \\
        --deduplicate_locations \\
        --T_target 20 \\
        --drug ap5_nbqx:experimental_data/CDKL5_02.h5:experimental_data/Immediately_after_drug_11.h5:well000 \\
        --drug bicuculline:experimental_data/CDKL5_02.h5:experimental_data/Immediately_after_drug_11.h5:well001 \\
        --drug_threshold_mad 5.0 \\
        --drug_n_jobs 4

Backward compatibility
----------------------
The two underlying scripts retain their original CLIs; this script only *imports*
their functions. Run them standalone the same way you always did.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Make sibling scripts importable regardless of where this file is invoked from.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Local module imports (reuse, don't reimplement).
import create_experimental_target as cet  # noqa: E402
import extract_drug_effects as ede  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_experimental_target_with_drugs")


# ---------------------------------------------------------------------------
# Drug registry validation (mirrors active model's drug_perturbations.py)
# ---------------------------------------------------------------------------

# Path to the model's drug registry. The CLAUDE.md flags DIV21_FxHET as the
# active variant of the CDKL5 model family, so that's the source of truth.
_DRUG_REGISTRY_PATH = (
    _HERE.parent
    / "RBS_network_models"
    / "models"
    / "CDKL5_E6D_T2_C1_05212024"
    / "DIV21_FxHET"
    / "src"
    / "drug_perturbations.py"
)


def _load_drug_registry():
    """Import the active model's drug_perturbations module and return its registry.

    Returns the ``DRUG_REGISTRY`` dict, or ``None`` if the module can't be loaded
    (in which case the caller should warn but proceed -- the user may be running
    against a model variant we don't know about).
    """
    if not _DRUG_REGISTRY_PATH.exists():
        return None
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "drug_perturbations_active", str(_DRUG_REGISTRY_PATH)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "DRUG_REGISTRY", None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load DRUG_REGISTRY from %s: %s", _DRUG_REGISTRY_PATH, exc)
        return None


# ---------------------------------------------------------------------------
# Per-drug spec parsing
# ---------------------------------------------------------------------------

class DrugSpec:
    """Parsed ``NAME:BASELINE_RAW:DRUG_RAW:WELL`` tuple."""

    __slots__ = ("name", "baseline_raw", "drug_raw", "well")

    def __init__(self, name: str, baseline_raw: Path, drug_raw: Path, well: str):
        self.name = name
        self.baseline_raw = baseline_raw
        self.drug_raw = drug_raw
        self.well = well

    def __repr__(self):
        return (
            f"DrugSpec(name={self.name!r}, baseline_raw={self.baseline_raw!s}, "
            f"drug_raw={self.drug_raw!s}, well={self.well!r})"
        )


def parse_drug_spec(raw: str) -> DrugSpec:
    """Parse a single ``--drug NAME:BASELINE_RAW:DRUG_RAW:WELL`` value.

    Splits on ':' from the *right* for the well and from the *left* for the name,
    so absolute paths containing ':' (uncommon but possible) survive. We require
    exactly 4 parts though, so realistically users should avoid colons in paths.
    """
    parts = raw.split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"--drug value must have 4 colon-separated parts "
            f"(NAME:BASELINE_RAW:DRUG_RAW:WELL); got {len(parts)}: {raw!r}"
        )
    name, baseline_raw, drug_raw, well = parts
    if not name:
        raise argparse.ArgumentTypeError(f"Drug name is empty in: {raw!r}")
    if not well:
        raise argparse.ArgumentTypeError(f"Well id is empty in: {raw!r}")
    return DrugSpec(
        name=name,
        baseline_raw=Path(baseline_raw),
        drug_raw=Path(drug_raw),
        well=well,
    )


# ---------------------------------------------------------------------------
# Drug-stamping phase
# ---------------------------------------------------------------------------

def stamp_drug(
    baseline_h5: Path,
    drug: DrugSpec,
    threshold_mad: float,
    peak_sign: str,
    n_jobs: int,
    clip_seconds: float,
    overwrite: bool,
) -> None:
    """Run the extract_drug_effects pipeline for a single drug and write into baseline_h5."""
    logger.info(
        "Drug %r: pre=%s post=%s well=%s",
        drug.name, drug.baseline_raw, drug.drug_raw, drug.well,
    )

    for p in (drug.baseline_raw, drug.drug_raw):
        if not p.exists():
            raise FileNotFoundError(f"Raw recording not found for drug {drug.name!r}: {p}")

    pre_units, post_units, source_meta = ede.load_from_raw_recordings(
        baseline_raw=drug.baseline_raw.resolve(),
        drug_raw=drug.drug_raw.resolve(),
        well=drug.well,
        threshold_mad=threshold_mad,
        peak_sign=peak_sign,
        n_jobs=n_jobs,
        clip_seconds=clip_seconds,
    )

    # Comparison window: prefer T_target_s already stamped into the baseline h5,
    # else fall back to min(pre_dur, post_dur). Matches extract_drug_effects.main.
    T = ede.get_T_from_baseline_h5(baseline_h5)
    if T is None:
        durations = []
        if pre_units:
            durations.append(max(t.max() for t in pre_units.values() if len(t)))
        if post_units:
            durations.append(max(t.max() for t in post_units.values() if len(t)))
        T = float(min(durations)) if durations else 0.0
        logger.warning(
            "No T_target_s in baseline h5 attrs; defaulting to min recording duration: %.2f s",
            T,
        )
    logger.info("Drug %r: comparison window T = %.2f s", drug.name, T)

    ratios, diagnostics = ede.compute_ratios(pre_units, post_units, T)

    logger.info("Drug %r: computed ratios:", drug.name)
    for k, v in ratios.items():
        logger.info("  %-30s = %.4f", k, v)

    extra_attrs = {
        "source_method": source_meta["source"],
        **{f"source_{k}": v for k, v in source_meta.items() if k != "source"},
    }
    ede.write_to_baseline_h5(
        baseline_h5=baseline_h5,
        drug_name=drug.name,
        ratios=ratios,
        diagnostics=diagnostics,
        extra_attrs=extra_attrs,
        overwrite=overwrite,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Baseline-target args (reuses the exact flag set from create_experimental_target).
    cet.add_baseline_target_arguments(parser)

    # Drug-stamping args.
    parser.add_argument(
        "--drug",
        action="append",
        default=[],
        metavar="NAME:BASELINE_RAW:DRUG_RAW:WELL",
        help=(
            "Repeatable. Each value is a colon-separated 4-tuple specifying one "
            "drug condition to stamp into the baseline h5. Different drugs can "
            "come from different (recording, well) pairs."
        ),
    )
    parser.add_argument(
        "--drug_threshold_mad",
        type=float,
        default=5.0,
        help="MAD-units detection threshold for raw MUA extraction (default 5.0).",
    )
    parser.add_argument(
        "--drug_peak_sign",
        choices=("neg", "pos", "both"),
        default="neg",
        help="Peak sign for detection (default neg).",
    )
    parser.add_argument(
        "--drug_n_jobs",
        type=int,
        default=1,
        help="Parallel workers for spikeinterface detect_peaks (default 1; raise to ~4 on a compute node).",
    )
    parser.add_argument(
        "--drug_clip_seconds",
        type=float,
        default=None,
        help=(
            "Time-slice both raw recordings to this many seconds before detection "
            "(big speedup when only the first T_target seconds will be analysed). "
            "Default: T_target + 5 if --T_target is set, else None (no clipping)."
        ),
    )
    parser.add_argument(
        "--drug_no_overwrite",
        action="store_true",
        help="Refuse to overwrite an existing /drug_effects/<drug_name> group in the h5.",
    )
    parser.add_argument(
        "--skip_registry_validation",
        action="store_true",
        help="Skip checking drug names against the model's DRUG_REGISTRY (use only when adding a new drug whose registry entry isn't merged yet).",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Parse and validate drug specs UP FRONT, before any heavy work.
    drug_specs = [parse_drug_spec(d) for d in args.drug]

    if drug_specs and not args.skip_registry_validation:
        registry = _load_drug_registry()
        if registry is None:
            logger.warning(
                "Could not load DRUG_REGISTRY from %s -- skipping drug-name validation. "
                "Pass --skip_registry_validation to silence this warning.",
                _DRUG_REGISTRY_PATH,
            )
        else:
            unknown = [d.name for d in drug_specs if d.name not in registry]
            if unknown:
                parser.error(
                    f"Unknown drug name(s): {unknown}. "
                    f"Registered drugs: {sorted(registry.keys())}. "
                    f"(Pass --skip_registry_validation to bypass.)"
                )
            logger.info(
                "Drug-registry validation OK for %d drug(s): %s",
                len(drug_specs), [d.name for d in drug_specs],
            )

    # Compute default clip_seconds from --T_target if user didn't override.
    clip_seconds = args.drug_clip_seconds
    if clip_seconds is None and args.T_target is not None:
        clip_seconds = float(args.T_target) + 5.0
        logger.info(
            "drug_clip_seconds auto-set to T_target + 5 = %.2f s "
            "(prevents detecting peaks past the analysis window).",
            clip_seconds,
        )

    overall_t0 = time.perf_counter()

    # ---- Phase 1: build baseline target h5 ----
    logger.info("=" * 70)
    logger.info("Phase 1/%d: building baseline target h5 -> %s", 1 + len(drug_specs), args.output)
    logger.info("=" * 70)
    phase_t0 = time.perf_counter()
    cet.build_baseline_target(args)
    logger.info(
        "Phase 1 done in %.1fs. Baseline target at %s",
        time.perf_counter() - phase_t0, args.output,
    )

    baseline_h5 = Path(args.output).resolve()
    if not baseline_h5.exists():
        # Defensive: should never happen, build_baseline_target should error out first.
        raise RuntimeError(f"Baseline target h5 was not created at {baseline_h5}")

    # ---- Phase 2..N: stamp each drug ----
    for i, drug in enumerate(drug_specs, start=2):
        logger.info("=" * 70)
        logger.info(
            "Phase %d/%d: stamping drug %r into %s",
            i, 1 + len(drug_specs), drug.name, baseline_h5,
        )
        logger.info("=" * 70)
        phase_t0 = time.perf_counter()
        stamp_drug(
            baseline_h5=baseline_h5,
            drug=drug,
            threshold_mad=args.drug_threshold_mad,
            peak_sign=args.drug_peak_sign,
            n_jobs=args.drug_n_jobs,
            clip_seconds=clip_seconds,
            overwrite=not args.drug_no_overwrite,
        )
        logger.info(
            "Phase %d done in %.1fs (drug=%r)",
            i, time.perf_counter() - phase_t0, drug.name,
        )

    logger.info("=" * 70)
    logger.info(
        "All phases complete in %.1fs total. Output: %s (%d drug(s) stamped)",
        time.perf_counter() - overall_t0, baseline_h5, len(drug_specs),
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
