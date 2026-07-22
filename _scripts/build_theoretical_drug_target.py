"""Build a THEORETICAL drug-effect target instead of extracting one from recordings.

``extract_drug_effects.py`` derives ``/drug_effects/<drug>/*_ratio`` from a paired
pre/post MEA recording. That is the right thing for fitting real data, but it is a
poor debugging probe: the ratios in the well currently being fit
(``CDKL5_002_well001.h5``) are close to 1.0 -- e.g. ``pop_FR_ratio = 1.015`` -- so
even a perfectly working drug-response term has almost no gradient to follow, and
a broken one is indistinguishable from a working one.

This script writes the same h5 structure, but with ratios derived from
pharmacological theory: large, unambiguous, directionally-correct effects. A run
against a theoretical target answers "does the drug term steer the search at all?"
which a run against a near-unity experimental target cannot.

The baseline part of the target (the ``unit_*`` groups and
``network_results.attrs['T_target_s']``) is copied through byte-for-byte, so
baseline fitting is completely unaffected -- only the drug ratios differ.

Ratio names must match the h5 dataset names in
``fitnessFunc_v2._DRUG_RATIO_FEATURE_MAP``; a name present in the fit schema but
absent here is silently skipped at fitness time, which quietly shrinks the
effective drug weight.

Usage
-----
    python build_theoretical_drug_target.py \
        --baseline_target processed_experimental_targets/CDKL5_002_well001.h5 \
        --output          processed_experimental_targets/CDKL5_002_well001_theory_bicuculline.h5 \
        --drug_name       bicuculline
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict

import h5py

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_theoretical_drug_target")


# ---------------------------------------------------------------------------
# Theoretical post/pre ratios
#
# ``max_dev`` is the deviation at which a metric contributes its full (= 1.0)
# loss in fitnessFunc_v2.compute_drug_response_score. It is set here by the rule
#
#     max_dev = |ratio - 1.0|
#
# so that a simulated ratio of exactly 1.0 -- "the drug did nothing" -- incurs
# precisely full loss, and any movement toward the target reduces loss linearly.
# That keeps every metric's normalized error on the same footing regardless of
# how large its effect is, and makes "no response" the well-defined worst case.
# ---------------------------------------------------------------------------

THEORETICAL_RATIOS: Dict[str, Dict[str, Dict[str, object]]] = {
    # Bicuculline: competitive GABA-A antagonist, modelled as scale_GABA = 0.0
    # (see src/drug_perturbations.py). Removing all fast inhibition raises firing
    # across the board and lets network bursts run long and recruit hard.
    "bicuculline": {
        "pop_FR_ratio": {
            "ratio": 2.0, "max_dev": 1.0,
            "rationale": "Loss of all GABA-A inhibition roughly doubles population firing.",
        },
        "exc_firing_rate_ratio": {
            "ratio": 2.2, "max_dev": 1.2,
            "rationale": "E cells lose their entire inhibitory drive; largest relative increase.",
        },
        "inh_firing_rate_ratio": {
            "ratio": 1.5, "max_dev": 0.5,
            "rationale": "I cells lose I->I inhibition and receive more E drive, so they speed up "
                         "too -- they simply no longer inhibit anything downstream.",
        },
        "network_burst_rate_ratio": {
            "ratio": 1.8, "max_dev": 0.8,
            "rationale": "More frequent network bursts; matches the direction measured in the "
                         "CDKL5 wells (well000 2.15, well001 1.56).",
        },
        "network_burst_duration_ratio": {
            "ratio": 3.0, "max_dev": 2.0,
            "rationale": "HEADLINE EFFECT. Without the GABA-A brake that terminates a burst, "
                         "network bursts run several times longer.",
        },
        "network_burst_amp_ratio": {
            "ratio": 2.5, "max_dev": 1.5,
            "rationale": "Disinhibition recruits far more units into each burst, so the peak "
                         "population rate rises sharply.",
        },
        "burstlet_duration_ratio": {
            "ratio": 2.0, "max_dev": 1.0,
            "rationale": "Same lengthening at the finer burstlet level, somewhat weaker.",
        },
        "burstlet_amp_ratio": {
            "ratio": 2.0, "max_dev": 1.0,
            "rationale": "Same recruitment increase at the finer burstlet level.",
        },
    },
}

# Deliberately excluded, and why -- recorded in the h5 so the choice is auditable.
EXCLUDED_FEATURES: Dict[str, Dict[str, str]] = {
    "bicuculline": {
        "superburst_rate_ratio": "Experimental value is 0.0 (no superbursts post-drug), so the "
                                 "ratio carries no gradient. The scorer also already max-penalises "
                                 "early superbursts independently.",
        "superburst_duration_ratio": "Same reason as superburst_rate_ratio.",
        "pre_burstlet_rate_ratio": "Redundant with burstlet_rate at a permissive threshold.",
        "pre_burstlet_duration_ratio": "Redundant with burstlet_duration.",
        "mean_participation_ratio": "Strongly correlated with burst amplitude; including both "
                                    "double-counts the same recruitment effect.",
    },
}


def _load_drug_registry():
    """Import DRUG_REGISTRY so theoretical targets can only be built for real drugs."""
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.src.drug_perturbations import (
        DRUG_REGISTRY,
    )
    return DRUG_REGISTRY


def write_theoretical_drug_effects(
    target_h5: Path,
    drug_name: str,
    spec: Dict[str, Dict[str, object]],
    registry_entry: Dict,
    source_target: Path,
    overwrite: bool,
    prune_other_drugs: bool = True,
) -> None:
    """Stamp /drug_effects/<drug_name> with theory-derived ratios.

    Mirrors extract_drug_effects.write_to_baseline_h5 so the two kinds of target
    are structurally interchangeable at fitness time -- the only difference is the
    provenance attrs, which mark this one as theoretical.

    ``prune_other_drugs`` drops any drug groups inherited from the copied source.
    Those are *measured* ratios, and leaving them in a file whose name says
    "theory" is a trap: fitting a second drug would silently mix a theoretical
    target for one drug with an experimental one for another. Dropping them makes
    such a run fail loudly with "no /drug_effects/<drug>" instead.
    """
    with h5py.File(target_h5, "a") as f:
        root = f.require_group("drug_effects")
        if prune_other_drugs:
            for other in [k for k in root if k != drug_name]:
                logger.warning(
                    "Dropping inherited experimental /drug_effects/%s "
                    "(this file holds theoretical ratios only)", other,
                )
                del root[other]
        if drug_name in root:
            if not overwrite:
                raise RuntimeError(
                    f"/drug_effects/{drug_name} already exists in {target_h5}. "
                    "Pass --overwrite to replace it."
                )
            logger.warning("Replacing existing /drug_effects/%s", drug_name)
            del root[drug_name]
        grp = root.create_group(drug_name)

        for feat, cfg in spec.items():
            grp.create_dataset(feat, data=float(cfg["ratio"]))

        # Diagnostics: the max_dev each ratio was designed against, plus the
        # reasoning. Lets the schema and the target be cross-checked after a run.
        diag = grp.create_group("diagnostics")
        dev_grp = diag.create_group("max_dev")
        for feat, cfg in spec.items():
            dev_grp.create_dataset(feat, data=float(cfg["max_dev"]))
        diag.attrs["max_dev_rule"] = (
            "max_dev = |ratio - 1.0|, so a simulated ratio of 1.0 (no drug response) "
            "incurs exactly full loss for that metric."
        )
        diag.attrs["rationale"] = json.dumps(
            {feat: cfg["rationale"] for feat, cfg in spec.items()}, indent=2
        )
        diag.attrs["excluded_features"] = json.dumps(
            EXCLUDED_FEATURES.get(drug_name, {}), indent=2
        )

        grp.attrs["drug_name"] = drug_name
        grp.attrs["source_method"] = "theoretical"
        grp.attrs["generation_timestamp"] = _dt.datetime.now().isoformat(timespec="seconds")
        grp.attrs["source_baseline_target"] = str(source_target)
        grp.attrs["cfg_overrides"] = json.dumps(registry_entry.get("cfg_overrides", {}))
        grp.attrs["drug_description"] = registry_entry.get("description", "")
        grp.attrs["warning"] = (
            "THEORETICAL ratios -- derived from pharmacology, NOT measured from a "
            "recording. Do not report these as experimental results."
        )

    logger.info("Wrote theoretical /drug_effects/%s into %s", drug_name, target_h5)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--baseline_target", required=True,
                        help="Existing baseline target h5 to copy (its unit_* groups and "
                             "T_target_s are preserved verbatim).")
    parser.add_argument("--output", required=True,
                        help="Path for the new theoretical target h5.")
    parser.add_argument("--drug_name", required=True,
                        help=f"Drug to stamp. Available: {sorted(THEORETICAL_RATIOS)}")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing /drug_effects/<drug> group, and an "
                             "existing output file.")
    parser.add_argument("--keep_other_drugs", action="store_true",
                        help="Keep drug groups inherited from --baseline_target. Off by "
                             "default: those hold MEASURED ratios, and mixing them into a "
                             "theoretical target silently mixes the two kinds of fit.")
    args = parser.parse_args()

    src = Path(args.baseline_target)
    dst = Path(args.output)
    drug = args.drug_name

    if not src.is_file():
        parser.error(f"--baseline_target not found: {src}")

    if drug not in THEORETICAL_RATIOS:
        parser.error(
            f"No theoretical ratio set defined for {drug!r}. "
            f"Available: {sorted(THEORETICAL_RATIOS)}. Add one to THEORETICAL_RATIOS."
        )

    registry = _load_drug_registry()
    if drug not in registry:
        parser.error(
            f"{drug!r} is not in DRUG_REGISTRY, so no simulation could apply it. "
            f"Registered: {sorted(registry)}"
        )

    if dst.exists() and not args.overwrite:
        parser.error(f"--output already exists: {dst}. Pass --overwrite to replace it.")

    if src.resolve() == dst.resolve():
        parser.error(
            "--output must differ from --baseline_target; the experimental target "
            "must stay intact so theory and experiment can be compared later."
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Copying baseline target %s -> %s", src, dst)
    shutil.copy2(src, dst)

    spec = THEORETICAL_RATIOS[drug]
    write_theoretical_drug_effects(
        target_h5=dst,
        drug_name=drug,
        spec=spec,
        registry_entry=registry[drug],
        source_target=src,
        overwrite=True,   # we just created the copy; any group in it came from the source
        prune_other_drugs=not args.keep_other_drugs,
    )

    print()
    print(f"Theoretical target for {drug}: {dst}")
    print(f"{'feature':<32} {'ratio':>7} {'max_dev':>8}")
    print("-" * 50)
    for feat, cfg in spec.items():
        print(f"{feat:<32} {float(cfg['ratio']):>7.2f} {float(cfg['max_dev']):>8.2f}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
