"""Extract experimental drug-response ratios and stamp them into a baseline target h5.

The optimization fitness compares simulated drug/baseline ratios to experimental
drug/baseline ratios for a small set of population-level rate features. This
script computes the experimental side: given a baseline target h5 (already
produced by create_experimental_target.py) and a paired pre/post drug
recording, it computes pre and post population rates + hierarchical burst
rates, takes the post/pre ratio, and writes the result into a new
/drug_effects/<drug_name>/ group inside the baseline h5.

Two input paths are supported:

  --baseline_raw + --drug_raw + --well            (raw extraction)
    Read both recordings as Maxwell h5 via spikeinterface, apply identical
    highpass + local CMR preprocessing (mirroring spikesort_drug_comparison.py),
    detect threshold-crossings on the channel intersection of pre & post.
    Noise levels are computed on PRE and reused on POST so the µV threshold
    is identical — drug effect manifests as rate change, not threshold drift.
    No spike-sorted data is read; pure MUA on both sides.

  --from_existing_results <well_dir>              (skip recompute)
    Read pre/post threshold-crossings already produced by
    MEA_Analysis/IPNAnalysis/spikesort_drug_comparison.py from the well's
    pre_post_thresh_crossings.npz (extremum-channel filtered; baseline analyzer
    was used to pick channels).

Population-firing-rate is computed as total_spikes / (n_units * T_seconds).
Hierarchical burst rates come from
MEA_Analysis.IPNAnalysis.parameter_free_burst_detector.compute_network_bursts
(production params: base_threshold_static=40, min_burstlet_participation=0.05).
Ratios use an epsilon floor of 1e-9 in the denominator to avoid blowups.
"""

import argparse
import datetime as _dt
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import h5py

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent  # .../networkSimulations
sys.path.insert(0, str(_REPO_ROOT / "MEA_Analysis" / "IPNAnalysis"))
from parameter_free_burst_detector import compute_network_bursts  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("extract_drug_effects")

EPS = 1e-9
# Main burst detector kwargs (matches fitnessFunc_v2._run_burst_detector main pass)
MAIN_BURST_KW = dict(base_threshold_static=40, min_burstlet_participation=0.05)
# Pre-burstlet detector kwargs (lower threshold; matches schema_v3 pre_burstlets component)
PRE_BURST_KW = dict(base_threshold_static=15, min_burstlet_participation=0.05)
# 80th percentile of firing rate -> 'inhibitory' (matches create_experimental_target.classify_units).
INHIB_QUANTILE = 0.80

RATIO_FEATURES = (
    "pop_FR_ratio",
    "exc_firing_rate_ratio",
    "inh_firing_rate_ratio",
    "burstlet_rate_ratio",
    "burstlet_duration_ratio",
    "network_burst_rate_ratio",
    "network_burst_duration_ratio",
    "superburst_rate_ratio",
    "superburst_duration_ratio",
    "pre_burstlet_rate_ratio",
    "pre_burstlet_duration_ratio",
    "mean_participation_ratio",
)


def _safe_ratio(num: float, denom: float) -> float:
    return float(num) / max(float(denom), EPS)


def _truncate(units: Dict[str, np.ndarray], T: float) -> Dict[str, np.ndarray]:
    return {u: t[t <= T] for u, t in units.items()}


def _pop_firing_rate_hz(units: Dict[str, np.ndarray], T: float) -> float:
    if not units or T <= 0:
        return 0.0
    total_spikes = sum(len(t) for t in units.values())
    return total_spikes / (len(units) * T)


def _per_channel_firing_rates_hz(
    units: Dict[str, np.ndarray], T: float
) -> Dict[str, float]:
    if T <= 0 or not units:
        return {}
    return {ch: len(t) / T for ch, t in units.items()}


def classify_channels_top20(
    pre_units: Dict[str, np.ndarray], T: float
) -> Tuple[Dict[str, str], float]:
    """Top INHIB_QUANTILE-th-percentile firing rate channels are inhibitory; rest excitatory.

    Mirrors create_experimental_target.classify_units (line 73-94) but operates on
    raw-MUA channel firing rates measured within the same window the ratios are
    computed on. Returns (classification, threshold_hz).
    """
    rates = _per_channel_firing_rates_hz(pre_units, T)
    if not rates:
        return {}, 0.0
    threshold = float(np.quantile(list(rates.values()), INHIB_QUANTILE))
    classification = {
        ch: ("inhibitory" if r >= threshold else "excitatory")
        for ch, r in rates.items()
    }
    return classification, threshold


def _class_mean_firing_rate_hz(
    units: Dict[str, np.ndarray], classification: Dict[str, str], cell_type: str, T: float
) -> float:
    """Mean firing rate (Hz) over channels of the given cell_type."""
    if T <= 0:
        return 0.0
    members = [ch for ch, c in classification.items() if c == cell_type and ch in units]
    if not members:
        return 0.0
    rates = [len(units[ch]) / T for ch in members]
    return float(np.mean(rates))


def _level_rate(nb_result: dict, level: str) -> float:
    return float(nb_result.get(level, {}).get("metrics", {}).get("rate", 0.0))


def _level_duration_mean(nb_result: dict, level: str) -> float:
    return float(nb_result.get(level, {}).get("metrics", {}).get("duration", {}).get("mean", 0.0))


def _level_participation_mean(nb_result: dict, level: str) -> float:
    return float(nb_result.get(level, {}).get("metrics", {}).get("participation", {}).get("mean", 0.0))


def load_from_existing_results(well_dir: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], dict]:
    """Read pre/post MUA threshold-crossings from spikesort_drug_comparison output."""
    npz_path = well_dir / "pre_post_thresh_crossings.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Expected {npz_path} (from spikesort_drug_comparison.py)")
    data = np.load(npz_path, allow_pickle=True)
    pre_units = {k.split("pre__", 1)[1]: data[k] for k in data.files if k.startswith("pre__")}
    post_units = {k.split("post__", 1)[1]: data[k] for k in data.files if k.startswith("post__")}
    if not pre_units or not post_units:
        raise RuntimeError(f"No pre__* / post__* arrays in {npz_path}")
    metadata = {"source": "from_existing_results", "well_dir": str(well_dir)}
    report_path = well_dir / "report.json"
    if report_path.exists():
        with open(report_path) as f:
            metadata["report"] = json.load(f)
    return pre_units, post_units, metadata


def load_from_raw_recordings(
    baseline_raw: Path,
    drug_raw: Path,
    well: str,
    threshold_mad: float,
    peak_sign: str,
    n_jobs: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], dict]:
    """Run the same preprocessing + threshold-detection chain as
    spikesort_drug_comparison.py, but on the channel intersection (no extremum
    filter from a sorted analyzer). Returns ({chan_id: spike_times_s}, ..., metadata)."""
    # Maxwell HDF5 compression plugin: h5py needs HDF5_PLUGIN_PATH to point at
    # libcompression.so before any read. Auto-set if not provided so the user
    # doesn't have to remember an env var.
    if "HDF5_PLUGIN_PATH" not in os.environ:
        for candidate in (
            "/global/homes/k/ktub1999/hdf5_plugin_path_maxwell",
            os.path.expanduser("~/hdf5_plugin_path_maxwell"),
        ):
            if os.path.isfile(os.path.join(candidate, "libcompression.so")):
                os.environ["HDF5_PLUGIN_PATH"] = candidate
                logger.info("HDF5_PLUGIN_PATH auto-set to %s", candidate)
                break
        else:
            logger.warning(
                "HDF5_PLUGIN_PATH not set and Maxwell libcompression.so not found "
                "in known locations; raw recording reads may fail."
            )

    sys.path.insert(0, str(_REPO_ROOT / "MEA_Analysis" / "IPNAnalysis"))
    import spikeinterface.full as si  # noqa: E402
    from spikesort_drug_comparison import (  # noqa: E402
        preprocess_recording,
        detect_threshold_crossings,
        peaks_by_channel,
        _select_channels,
    )

    logger.info("[raw] reading baseline %s (stream=%s)", baseline_raw, well)
    pre_rec = si.read_maxwell(str(baseline_raw), stream_id=well, rec_name="rec0000")
    pre_rec = preprocess_recording(pre_rec, logger)
    logger.info("[raw] reading drug    %s (stream=%s)", drug_raw, well)
    post_rec = si.read_maxwell(str(drug_raw), stream_id=well, rec_name="rec0000")
    post_rec = preprocess_recording(post_rec, logger)

    # Channel intersection so pre and post are scored on the same electrodes.
    pre_ids = [str(c) for c in pre_rec.get_channel_ids()]
    post_ids = [str(c) for c in post_rec.get_channel_ids()]
    common = [c for c in pre_ids if c in set(post_ids)]
    if not common:
        raise RuntimeError("No shared channels between pre and post recordings")
    logger.info(
        "[raw] channel intersection: %d (pre had %d, post had %d)",
        len(common), len(pre_ids), len(post_ids),
    )
    rec_ids_native = list(pre_rec.get_channel_ids())
    if rec_ids_native and not isinstance(rec_ids_native[0], str):
        cast = type(rec_ids_native[0])
        common_native = [cast(c) for c in common]
    else:
        common_native = common
    pre_sliced = _select_channels(pre_rec, common_native)
    post_sliced = _select_channels(post_rec, common_native)

    logger.info("[raw] computing per-channel noise on pre (sliced)...")
    noise = np.asarray(si.get_noise_levels(pre_sliced, return_scaled=False), dtype=np.float32)
    logger.info(
        "[raw] pre noise: median=%.3f  min=%.3f  max=%.3f",
        float(np.median(noise)), float(noise.min()), float(noise.max()),
    )

    fs = float(pre_sliced.get_sampling_frequency())
    pre_dur = float(pre_sliced.get_duration())
    post_dur = float(post_sliced.get_duration())

    peaks_pre = detect_threshold_crossings(
        pre_sliced, noise, threshold_mad, peak_sign, n_jobs, logger, "pre",
    )
    peaks_post = detect_threshold_crossings(
        post_sliced, noise, threshold_mad, peak_sign, n_jobs, logger, "post",
    )
    pre_units = peaks_by_channel(peaks_pre, common, fs)
    post_units = peaks_by_channel(peaks_post, common, fs)

    metadata = {
        "source": "raw_recordings",
        "baseline_raw": str(baseline_raw),
        "drug_raw": str(drug_raw),
        "well": well,
        "threshold_mad": float(threshold_mad),
        "peak_sign": peak_sign,
        "n_channels_intersect": len(common),
        "pre_recording_seconds": pre_dur,
        "post_recording_seconds": post_dur,
        "fs_hz": fs,
        "noise_median": float(np.median(noise)),
    }
    return pre_units, post_units, metadata


def get_T_from_baseline_h5(baseline_h5: Path) -> Optional[float]:
    with h5py.File(baseline_h5, "r") as f:
        if "network_results" not in f:
            return None
        attrs = dict(f["network_results"].attrs)
        for k in ("T_target_s", "T_seconds", "T_target", "duration_seconds"):
            if k in attrs:
                return float(attrs[k])
    return None


def _side_features(
    units: Dict[str, np.ndarray],
    T: float,
    classification: Dict[str, str],
    label: str,
) -> dict:
    """Run main + pre-burstlet detectors and return all the rate/duration features
    we'll ratio (matches the schema_v3 metric coverage that's safe to ratio:
    rates, durations, mean participation, plus E/I-split mean firing rates)."""
    pop_FR = _pop_firing_rate_hz(units, T)
    exc_FR = _class_mean_firing_rate_hz(units, classification, "excitatory", T)
    inh_FR = _class_mean_firing_rate_hz(units, classification, "inhibitory", T)
    logger.info(
        "[%s] pop_FR=%.4f Hz  exc_FR=%.4f Hz  inh_FR=%.4f Hz  (%d units)",
        label, pop_FR, exc_FR, inh_FR, len(units),
    )

    logger.info("[%s] running main burst detector (base_threshold_static=40)...", label)
    main_nb = compute_network_bursts(SpikeTimes=units, plot=False, verbose=False, **MAIN_BURST_KW)
    if "error" in main_nb:
        raise RuntimeError(f"{label}-side main burst detector failed: {main_nb['error']}")

    logger.info("[%s] running pre-burstlet detector (base_threshold_static=15)...", label)
    pre_nb = compute_network_bursts(SpikeTimes=units, plot=False, verbose=False, **PRE_BURST_KW)
    if "error" in pre_nb:
        raise RuntimeError(f"{label}-side pre-burstlet detector failed: {pre_nb['error']}")

    return {
        "pop_FR_hz": pop_FR,
        "exc_firing_rate_hz": exc_FR,
        "inh_firing_rate_hz": inh_FR,
        # main pass
        "burstlet_rate_hz":          _level_rate(main_nb, "burstlets"),
        "burstlet_duration_s":       _level_duration_mean(main_nb, "burstlets"),
        "network_burst_rate_hz":     _level_rate(main_nb, "network_bursts"),
        "network_burst_duration_s":  _level_duration_mean(main_nb, "network_bursts"),
        "superburst_rate_hz":        _level_rate(main_nb, "superbursts"),
        "superburst_duration_s":     _level_duration_mean(main_nb, "superbursts"),
        "mean_participation":        _level_participation_mean(main_nb, "network_bursts"),
        # pre-burstlet pass (permissive threshold)
        "pre_burstlet_rate_hz":      _level_rate(pre_nb, "burstlets"),
        "pre_burstlet_duration_s":   _level_duration_mean(pre_nb, "burstlets"),
    }


def compute_ratios(
    pre_units: Dict[str, np.ndarray],
    post_units: Dict[str, np.ndarray],
    T: float,
) -> Tuple[dict, dict]:
    """Return (ratios_dict, diagnostics_dict).

    Channel E/I classification is derived from the PRE side only (top-20% firing
    rate -> inhibitory) and applied identically to the POST side, so a channel's
    label is fixed by its baseline behaviour. Mean firing rates per class are
    computed on the same channel set in both conditions.
    """
    pre_clip = _truncate(pre_units, T)
    post_clip = _truncate(post_units, T)

    classification, ei_threshold_hz = classify_channels_top20(pre_clip, T)
    n_inh = sum(1 for c in classification.values() if c == "inhibitory")
    n_exc = sum(1 for c in classification.values() if c == "excitatory")
    logger.info(
        "E/I split (pre top-%.0f%% firing rate, threshold=%.4f Hz): %d inh, %d exc",
        (1.0 - INHIB_QUANTILE) * 100, ei_threshold_hz, n_inh, n_exc,
    )

    pre = _side_features(pre_clip, T, classification, "pre")
    post = _side_features(post_clip, T, classification, "post")

    ratios = {
        "pop_FR_ratio":                   _safe_ratio(post["pop_FR_hz"], pre["pop_FR_hz"]),
        "exc_firing_rate_ratio":          _safe_ratio(post["exc_firing_rate_hz"], pre["exc_firing_rate_hz"]),
        "inh_firing_rate_ratio":          _safe_ratio(post["inh_firing_rate_hz"], pre["inh_firing_rate_hz"]),
        "burstlet_rate_ratio":            _safe_ratio(post["burstlet_rate_hz"], pre["burstlet_rate_hz"]),
        "burstlet_duration_ratio":        _safe_ratio(post["burstlet_duration_s"], pre["burstlet_duration_s"]),
        "network_burst_rate_ratio":       _safe_ratio(post["network_burst_rate_hz"], pre["network_burst_rate_hz"]),
        "network_burst_duration_ratio":   _safe_ratio(post["network_burst_duration_s"], pre["network_burst_duration_s"]),
        "superburst_rate_ratio":          _safe_ratio(post["superburst_rate_hz"], pre["superburst_rate_hz"]),
        "superburst_duration_ratio":      _safe_ratio(post["superburst_duration_s"], pre["superburst_duration_s"]),
        "pre_burstlet_rate_ratio":        _safe_ratio(post["pre_burstlet_rate_hz"], pre["pre_burstlet_rate_hz"]),
        "pre_burstlet_duration_ratio":    _safe_ratio(post["pre_burstlet_duration_s"], pre["pre_burstlet_duration_s"]),
        "mean_participation_ratio":       _safe_ratio(post["mean_participation"], pre["mean_participation"]),
    }

    diagnostics = {
        "T_seconds": T,
        "n_pre_units": len(pre_clip),
        "n_post_units": len(post_clip),
        "n_inhibitory_channels": n_inh,
        "n_excitatory_channels": n_exc,
        "ei_split_quantile": INHIB_QUANTILE,
        "ei_threshold_firing_rate_hz": ei_threshold_hz,
        "pre_total_spikes": int(sum(len(t) for t in pre_clip.values())),
        "post_total_spikes": int(sum(len(t) for t in post_clip.values())),
        "pre": pre,
        "post": post,
        "main_burst_kwargs": MAIN_BURST_KW,
        "pre_burstlet_kwargs": PRE_BURST_KW,
        "classification": classification,
    }
    return ratios, diagnostics


def write_to_baseline_h5(
    baseline_h5: Path,
    drug_name: str,
    ratios: dict,
    diagnostics: dict,
    extra_attrs: dict,
    overwrite: bool,
) -> None:
    with h5py.File(baseline_h5, "a") as f:
        root = f.require_group("drug_effects")
        if drug_name in root:
            if not overwrite:
                raise RuntimeError(
                    f"/drug_effects/{drug_name} already exists in {baseline_h5}. "
                    "Re-run with --no_overwrite removed (or pick a different --drug_name)."
                )
            del root[drug_name]
        grp = root.create_group(drug_name)
        for k, v in ratios.items():
            grp.create_dataset(k, data=float(v))
        diag = grp.create_group("diagnostics")
        for side in ("pre", "post"):
            sgrp = diag.create_group(side)
            for k, v in diagnostics[side].items():
                sgrp.create_dataset(k, data=float(v))
        for k in (
            "T_seconds", "n_pre_units", "n_post_units",
            "n_inhibitory_channels", "n_excitatory_channels",
            "ei_split_quantile", "ei_threshold_firing_rate_hz",
            "pre_total_spikes", "post_total_spikes",
        ):
            diag.attrs[k] = diagnostics[k]
        diag.attrs["main_burst_kwargs"] = json.dumps(diagnostics["main_burst_kwargs"])
        diag.attrs["pre_burstlet_kwargs"] = json.dumps(diagnostics["pre_burstlet_kwargs"])

        # Persist E/I classification as two channel-id arrays so it can be
        # inspected and reused (e.g., by a fitness function that wants the same
        # channel split applied to a fresh recording).
        classification = diagnostics["classification"]
        inhib_chans = sorted(ch for ch, c in classification.items() if c == "inhibitory")
        excit_chans = sorted(ch for ch, c in classification.items() if c == "excitatory")
        cls_grp = diag.create_group("classification")
        str_dt = h5py.string_dtype(encoding="utf-8")
        cls_grp.create_dataset("inhibitory_channels", data=np.array(inhib_chans, dtype=str_dt))
        cls_grp.create_dataset("excitatory_channels", data=np.array(excit_chans, dtype=str_dt))
        cls_grp.attrs["rule"] = (
            f"top {(1.0 - INHIB_QUANTILE) * 100:.0f}% by pre-drug per-channel firing rate -> inhibitory; "
            "same labels applied to post-drug recording on the same channel ids"
        )

        grp.attrs["drug_name"] = drug_name
        grp.attrs["extraction_timestamp"] = _dt.datetime.now().isoformat(timespec="seconds")
        for k, v in extra_attrs.items():
            grp.attrs[k] = v if isinstance(v, (str, int, float, bool)) else json.dumps(v)
    logger.info("Wrote /drug_effects/%s into %s", drug_name, baseline_h5)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline_target", required=True, help="Path to existing baseline target h5 produced by create_experimental_target.py")
    parser.add_argument("--drug_name", required=True, help="Drug condition label (e.g., bicuculline). Becomes the /drug_effects/<name>/ group key.")
    parser.add_argument("--from_existing_results", help="Path to a well_dir containing pre_post_thresh_crossings.npz (output of spikesort_drug_comparison.py). Mutually exclusive with --baseline_raw / --drug_raw.")
    parser.add_argument("--baseline_raw", help="Path to baseline Maxwell .h5 raw recording (read via spikeinterface).")
    parser.add_argument("--drug_raw", help="Path to post-drug Maxwell .h5 raw recording.")
    parser.add_argument("--well", default="well000", help="Maxwell stream id for both recordings (default: well000).")
    parser.add_argument("--threshold_mad", type=float, default=5.0, help="Detection threshold in MAD units for raw extraction (default: 5.0; matches spikesort_drug_comparison.py).")
    parser.add_argument("--peak_sign", choices=("neg", "pos", "both"), default="neg", help="Peak sign for detection (default: neg).")
    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel workers for detect_peaks (default: 1; raise to ~4 on a compute node).")
    parser.add_argument("--T_seconds", type=float, default=None, help="Window length applied to both pre and post spike trains. Default: T_target_s from baseline h5; falls back to min(pre_dur, post_dur) if absent.")
    parser.add_argument("--no_overwrite", action="store_true", help="Refuse to overwrite an existing /drug_effects/<drug_name> group.")
    args = parser.parse_args()

    baseline_h5 = Path(args.baseline_target).resolve()
    if not baseline_h5.exists():
        parser.error(f"Baseline target h5 not found: {baseline_h5}")

    raw_provided = bool(args.baseline_raw or args.drug_raw)
    if raw_provided and args.from_existing_results:
        parser.error("--from_existing_results is mutually exclusive with --baseline_raw/--drug_raw.")
    if raw_provided and not (args.baseline_raw and args.drug_raw):
        parser.error("Both --baseline_raw and --drug_raw must be provided together.")

    if args.from_existing_results:
        well_dir = Path(args.from_existing_results).resolve()
        pre_units, post_units, source_meta = load_from_existing_results(well_dir)
    elif raw_provided:
        baseline_raw = Path(args.baseline_raw).resolve()
        drug_raw = Path(args.drug_raw).resolve()
        for p in (baseline_raw, drug_raw):
            if not p.exists():
                parser.error(f"Raw recording not found: {p}")
        pre_units, post_units, source_meta = load_from_raw_recordings(
            baseline_raw=baseline_raw,
            drug_raw=drug_raw,
            well=args.well,
            threshold_mad=args.threshold_mad,
            peak_sign=args.peak_sign,
            n_jobs=args.n_jobs,
        )
    else:
        parser.error("Provide either --baseline_raw + --drug_raw, or --from_existing_results <well_dir>.")

    T = args.T_seconds
    if T is None:
        T = get_T_from_baseline_h5(baseline_h5)
    if T is None:
        durations = []
        if pre_units:
            durations.append(max(t.max() for t in pre_units.values() if len(t)))
        if post_units:
            durations.append(max(t.max() for t in post_units.values() if len(t)))
        T = float(min(durations)) if durations else 0.0
        logger.warning("No T_target_s in baseline h5 attrs; defaulting to min recording duration: %.2f s", T)
    logger.info("Comparison window T = %.2f s", T)

    ratios, diagnostics = compute_ratios(pre_units, post_units, T)

    logger.info("Computed ratios:")
    for k, v in ratios.items():
        logger.info("  %-30s = %.4f", k, v)

    extra_attrs = {
        "source_method": source_meta["source"],
        **{f"source_{k}": v for k, v in source_meta.items() if k != "source"},
    }
    write_to_baseline_h5(
        baseline_h5=baseline_h5,
        drug_name=args.drug_name,
        ratios=ratios,
        diagnostics=diagnostics,
        extra_attrs=extra_attrs,
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()
