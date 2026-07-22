# Experimental Data Fitting — Pipeline Overview & Roadmap

This document explains **how this repository fits a biophysical NetPyNE/NEURON
network model to MEA recordings**, lists every feature currently used in the
fitness objective, and tracks what is implemented vs. still pending.

The active branch is `multi-drug-optimization` (in `RBS_network_models/`,
which is an independent git repo). Detailed phase-by-phase implementation
plan for the in-flight drug-response work lives in the linked plan:

- **Plan:** [`MULTI_DRUG_PLAN.md`](./MULTI_DRUG_PLAN.md)

---

## 1. High-level fitting strategy

Each evolutionary candidate is a **parameter vector** sampled by the
optimizer (Optuna TPE or inspyred EA). For every candidate we:

1. Override the relevant `simConfig` fields with the candidate's params.
2. Run a NetPyNE/NEURON network simulation (`sim.create → sim.simulate → sim.analyze`)
   inside `models/.../src/init.py`, driven by `batchEvol_v2` in `batch.py`.
3. Extract a set of **population- and unit-level features** from the resulting
   spike trains.
4. Compare those features against an **experimental target HDF5** built from
   spike-sorted MEA recordings.
5. Return a **weighted scalar fitness** (lower = better) used by the optimizer
   to propose the next candidate.

Active model + condition: `CDKL5_E6D_T2_C1_05212024/DIV21_FxHET`.
Active target: `processed_experimental_targets/CDKL5_002_well000.h5`.

Top-level launcher: `models/.../DIV21_FxHET/run_batch.py`
(currently imports `evol_params_from_seed_v3_large_v3`, `seeds_6`, and
`fitness_schema/schema_v3`).

---

## 2. Experimental target construction

**Script:** `_scripts/create_experimental_target.py`

Inputs (per well, post Kilosort + curation):
- `spike_times.npy` — `{unit_id: spike_times_seconds}` from spike sorter
- `metrics_curated.xlsx` — SpikeInterface quality metrics per unit
- `network_results.json` — network-level outputs from MEA pipeline

Behaviour:
- Loads spike times and quality metrics.
- Classifies units: **top 20% firing-rate units → inhibitory**, rest excitatory
  (the same `INHIB_QUANTILE = 0.80` is reused by `extract_drug_effects.py`).
- Stores each unit under `/unit_{id}/spike_times` (dataset) +
  per-unit attrs (`firing_rate`, `presence_ratio`, `snr`, ISI violation stats,
  amplitude statistics, `sync_spike_{2,4,8}`, `firing_range`, `sd_ratio`,
  `loc_x`, `loc_y`, `cell_type`, …).
- Writes top-level `network_results.attrs["T_target_s"]` — the **convergence
  window** within which features stabilize. Computed by
  `_scripts/feature_window.py`. This `T_target_s` is then propagated through
  the `T_TARGET_S` env var to `init.py` and used as the comparison window on
  both sides (sim and experiment).

---

## 3. Feature catalogue (what we currently fit on)

The authoritative list of metrics is in
`models/.../DIV21_FxHET/fitness_schema/schema_v3.py`. Features fall into six
component groups, each with its own weight in the total fitness. Per-metric
weights inside a component are normalized to sum to 1.

| Component | Weight | Metrics (with active per-metric weight) |
| --- | --- | --- |
| **`unit_metrics`** — per-unit spiking statistics | 0.50 | `firing_rate_error_exc` (0.10), `firing_rate_error_inh` (0.70), `num_spikes_error` (0.20), plus zero-weight slots for `firing_range_error`, `cv_isi_error`, `n_units_error`, `ei_ratio_error`, aggregate `firing_rate_error` |
| **`quality_metrics`** — waveform/QC proxies | 0.00 | `snr_error`, `amplitude_median_error`, `presence_ratio_error`, `isi_violations_error` *(group disabled)* |
| **`synchrony`** — spike-time coincidence | 0.00 | `sync_spike_{2,4,8}_error`, `pairwise_correlation_error` *(group disabled)* |
| **`burstlets`** — hierarchical level 1 (fast small bursts) | 0.20 | `burst_count_error` (0.45), `duration_error` (0.25), `timing_error` (0.30, Victor–Purpura on burstlet start times), plus zero-weight `burst_rate_error`, `participation_error`, `spikes_per_burst_error` |
| **`network_bursts`** — level 2 (merged medium-scale bursts) | 0.10 | `burst_count_error` (0.45), `duration_error` (0.25), `timing_error` (0.30), plus zero-weight `burst_rate_error`, `ibi_error`, `participation_error`, `intensity_error` |
| **`superbursts`** — level 3 (long clustered events) | 0.00 | `burst_rate_error`, `burst_count_error`, `duration_error`, `timing_error` *(group disabled)* |
| **`pre_burstlets`** — permissive-threshold burstlets | 0.20 | `burst_count_error` (0.45), `duration_error` (0.25), `timing_error` (0.30), plus zero-weight rate/participation/spikes-per-burst |

Engine: hierarchical burst features come from
`MEA_Analysis/IPNAnalysis/parameter_free_burst_detector.compute_network_bursts`.
Two detector passes run for every candidate:

- **Main pass:** `base_threshold_static=40`, `min_burstlet_participation=0.05`
  → feeds `burstlets`, `network_bursts`, `superbursts`.
- **Pre-burstlet pass:** `base_threshold_static=15`,
  `min_burstlet_participation=0.05` → feeds `pre_burstlets`.

Scoring engine is `the_scoring_function_asymmetric_parabola` (per metric
inside `fitnessFunc_v2.py`). Each error is clipped to its `[min_val, max_val]`
range from the schema, then squared/weighted. **Dealbreakers** (all-E firing,
all-I firing, neurons with zero synapses, etc.) short-circuit to
`max_fitness` so the optimizer learns to avoid those parameter basins.

---

## 4. Evolutionary feature add-ons (already in `multi-drug-optimization` branch)

These are the recent improvements layered on top of the base fitting flow.

### 4a. Dynamic `T_target_s` (commit `6ede39b46`)
`_scripts/feature_window.py` analyses the experimental spike trains for the
window after which population-level rate features stop drifting. The chosen
`T_target_s` is written into `network_results.attrs` of the target h5 and
plumbed to simulations via the `T_TARGET_S` env var, so both sides are
compared on the same biological window length. Diagnostics:
`CDKL5_002_well000_feature_convergence{.csv,.png,.pdf}`.

### 4b. Inhibitory/Excitatory firing-rate split
- Experimental side: `classify_units` in `create_experimental_target.py`
  labels the top 20% firing-rate units as inhibitory (`cell_type='I'`),
  rest as excitatory (`cell_type='E'`).
- Simulated side: `compute_unit_score` in `fitnessFunc_v2.py` rebuilds E/I
  labels from NetPyNE `popData`, then matches the same E and I mean-firing-
  rate targets.
- Schema v3 puts `0.70` on `firing_rate_error_inh` and `0.10` on
  `firing_rate_error_exc`, reflecting the biology where the inhibitory
  population is the harder-to-fit minority.

### 4c. Pre-burstlets (permissive-threshold layer)
Added so the optimizer feels gradient on weak/early burst structure. Same
metrics as `burstlets` but using threshold `15` instead of `40`. Currently
weighted equal to `burstlets` (0.20).

### 4d. Victor–Purpura timing distance
`_victor_purpura_distance` in `fitnessFunc_v2.py` is added to every burst
level (burstlets / network_bursts / superbursts / pre_burstlets) under
`timing_error`, providing a temporal-alignment signal on burst onset times
that count/duration metrics alone cannot supply.

### 4e. Crash-safety net (`init.py` rank-0 sentinel writer)
Wraps `sim.create / sim.simulate / sim.analyze` in `try/except BaseException`.
Any catchable NEURON failure now produces `<simLabel>_data.pkl` plus a
`<simLabel>_fitness.json` containing `max_fitness`. This replaces the old
behaviour where one diverging candidate would `MPI_Abort` all 128 ranks of
the trial and force the orchestrator into a `maxiter_wait * time_sleep`
timeout. Optuna now treats the failure as a worst-fitness datapoint and
moves on.

### 4f. Pharmacological scalers exposed (commits `ea70e4f26`, `35ddf6a7f`)
- `cfg.scale_AMPA`, `cfg.scale_NMDA`, `cfg.scale_GABA`, `cfg.scale_K` are
  defined in `cfg.py:116-119` and applied at `netParams.py:245-246, 265`.
- These were the prerequisite for the multi-drug work (Phase 1 + 2 of
  [`MULTI_DRUG_PLAN.md`](./MULTI_DRUG_PLAN.md)).

---

## 5. Multi-drug response fitting (Phases 1 + 2 done; 3–6 pending)

The new direction: rather than fit only baseline activity, **also fit how the
network responds to pharmacological perturbations** by comparing post/pre
ratios of population-level rate features between simulation and experiment.
Spike sorting the post-drug recording is unreliable (units drift), so all
drug-side features are **population-level**, MUA-based, and dimensionless
(ratios). Full design in [`MULTI_DRUG_PLAN.md`](./MULTI_DRUG_PLAN.md).

### Already done

#### Phase 1 — Experimental drug-effect ratio extraction
`_scripts/extract_drug_effects.py` (~520 LOC).

- Reads paired raw Maxwell `.h5` (or pre-computed
  `pre_post_thresh_crossings.npz` from
  `MEA_Analysis/IPNAnalysis/spikesort_drug_comparison.py`).
- Computes **12 post/pre ratios**:
  `pop_FR_ratio`, `exc_firing_rate_ratio`, `inh_firing_rate_ratio`,
  `burstlet_rate_ratio`, `burstlet_duration_ratio`,
  `network_burst_rate_ratio`, `network_burst_duration_ratio`,
  `superburst_rate_ratio`, `superburst_duration_ratio`,
  `pre_burstlet_rate_ratio`, `pre_burstlet_duration_ratio`,
  `mean_participation_ratio`.
- Uses the same `compute_network_bursts` params as `fitnessFunc_v2`
  (main: `base_threshold_static=40`, pre: `15`; both
  `min_burstlet_participation=0.05`).
- E/I split via the pre-drug top-20% firing-rate rule; **same channel
  labels** carry over to the post recording so the E/I assignment is
  consistent across conditions.
- Channel intersection between pre & post, pre-derived noise level reused on
  post → identical µV threshold; drug effect shows as rate change, not
  threshold drift.
- Writes to `/drug_effects/<drug_name>/` inside the baseline target h5
  (datasets per ratio + diagnostic absolute rates + attrs incl.
  `drug_param_overrides` JSON).
- Verified on `CDKL5_02` ↔ `Immediately_after_drug_11` well000 →
  bicuculline ratios stamped into
  `processed_experimental_targets/CDKL5_002_well000.h5`.

#### Phase 2 — Drug perturbation registry
`models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/src/drug_perturbations.py`.

`DRUG_REGISTRY` covers `bicuculline`, `gabazine`, `ap5_nbqx`, `ap5`,
`fourap`, each mapping to a `cfg_overrides` dict over the scalers exposed in
4f. Helpers: `validate_drug`, `get_overrides`, `apply_drug` (mutates cfg +
returns a snapshot tuple), `restore_cfg`. Same registry is consumed by both
`init.py` (live cfg mutation, future Phase 3) and
`extract_drug_effects.py` (records the param overrides into the target h5).

### Pending — Phases 3 through 6

| Phase | File(s) | Scope |
| --- | --- | --- |
| **3** — Two-sim init.py + batch.py | `src/init.py`, `src/batch.py` | Per-condition loop in `init.py` (baseline + one sim per drug listed in `os.environ["DRUGS"]`); save `drug_simData[drug] = {spkt, spkid}` into the trial pkl. `batch.py` reads `kwargs['drugs']`, validates against `DRUG_REGISTRY`, sets the env var, and forwards `drugs` into `fitnessFuncArgs`. Crash sentinel extended to cover all conditions. |
| **4** — Fitness `drug_response` component | `fitnessFunc_v2.py` | For each drug present in the trial pkl: compute baseline-sim and drug-sim `pop_FR`, network burst rate, burstlet rate, superburst rate over the same `T_target_s` window; form `sim_ratio = drug / max(baseline, 1e-9)`; compare against `/drug_effects/<drug>/` in the reference h5; per-feature loss = `|sim_ratio − exp_ratio|` scored against `max_dev`. Mean across drugs → single `drug_response` component score. |
| **5** — `schema_v3_drug` | `fitness_schema/schema_v3_drug.py` | Mirror `schema_v3` and add a `drug_response` component at weight ≈ 0.40, holding `pop_FR_ratio`, `network_burst_rate_ratio`, `burstlet_rate_ratio`, `superburst_rate_ratio` (initial weights/max_devs in the plan). Existing baseline components rescaled accordingly. |
| **6** — `--drugs` CLI flag in `run_batch.py` | `run_batch.py` | `--drugs bicuculline gabazine …`; when non-empty, swap active schema import to `schema_v3_drug`. Plumb `drugs=...` into `kwargs`. Early validation against `DRUG_REGISTRY`. |

Verification matrix (smoke test, baseline regression, failure isolation,
small Perlmutter batch, Optuna sanity) is enumerated at the bottom of
[`MULTI_DRUG_PLAN.md`](./MULTI_DRUG_PLAN.md).

### Out of scope (deliberately, for now)
- Acquiring more drug-pair recordings (only `CDKL5_02 ↔ Imm_11` today).
- Optimizer-learned drug intensities — drug → cfg-param map is **hand-coded**.
- E/I-resolved drug response — post-drug spike sorting is not trustworthy,
  so drug-side features stay population-level.
- Drug-onset transient dynamics — we model steady-state pre/post only.
- Drug effects on synaptic kinetics (`tau1_GABA`, `tau2_GABA`) — only
  conductance scaling for v1.

---

## 6. Quick map: where to look in the code

- Target builder: `_scripts/create_experimental_target.py`,
  helper `_scripts/feature_window.py`
- Drug-effect ratio extractor: `_scripts/extract_drug_effects.py`
- Drug registry: `models/.../DIV21_FxHET/src/drug_perturbations.py`
- Simulation entry: `models/.../DIV21_FxHET/src/init.py`
  (drives per-rank NEURON sim, writes `<simLabel>_data.pkl`)
- Batch driver: `models/.../DIV21_FxHET/src/batch.py`
  (NetPyNE `Batch` setup, env-var plumbing, fitnessFunc wiring)
- Active fitness function: `RBS_network_models/fitnessFunc_v2.py`
- Active schema: `models/.../DIV21_FxHET/fitness_schema/schema_v3.py`
- Pharmacological scalers: `models/.../DIV21_FxHET/src/cfg.py:110-119`,
  applied in `netParams.py:245-246, 265`
- Burst detector (external, reused on both experimental and simulated
  spike trains): `MEA_Analysis/IPNAnalysis/parameter_free_burst_detector.py`

---

**See also:** the in-flight multi-drug implementation plan
[`MULTI_DRUG_PLAN.md`](./MULTI_DRUG_PLAN.md) for phase-level details,
verification steps, and design rationale for the work pending in §5.
