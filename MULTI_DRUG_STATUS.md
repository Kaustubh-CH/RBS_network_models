# Multi-Drug Optimization — Implementation Status

Branch: `multi-drug-optimization` (in this repo, branched from `dev_branch`)

Plan: see `MULTI_DRUG_PLAN.md` in this directory.

## Done

### Phase 1 — `_scripts/extract_drug_effects.py`
Reads paired pre/post-drug recordings (raw Maxwell `.h5` or pre-computed
`pre_post_thresh_crossings.npz`), classifies channels (top-20 % pre-drug
firing rate → inhibitory, same labels applied post), and writes 12 ratio
features into `/drug_effects/<drug_name>/` of the baseline target h5.

Verified end-to-end on `experimental_data/CDKL5_02.h5` (well000) vs
`experimental_data/Immediately_after_drug_11.h5` (well000); ratios stamped
into `processed_experimental_targets/CDKL5_002_well000.h5` under
`/drug_effects/bicuculline/`.

Run command (canonical):
```bash
HDF5_PLUGIN_PATH=/global/homes/k/ktub1999/hdf5_plugin_path_maxwell \
/pscratch/sd/k/ktub1999/.conda/envs/preshifter/bin/python \
  RBS_network_models/_scripts/extract_drug_effects.py \
  --baseline_target processed_experimental_targets/CDKL5_002_well000.h5 \
  --baseline_raw    experimental_data/CDKL5_02.h5 \
  --drug_raw        experimental_data/Immediately_after_drug_11.h5 \
  --well well000 --drug_name bicuculline --n_jobs 1
```
(`--n_jobs 1` on login nodes; multi-worker triggers `BrokenProcessPool`.)

### Phase 2 — `models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/src/drug_perturbations.py`
`DRUG_REGISTRY` mapping drug name → cfg attribute overrides
(bicuculline / gabazine → `scale_GABA=0`, ap5_nbqx → `scale_AMPA=scale_NMDA=0`,
ap5 → `scale_NMDA=0`, fourap → `scale_K=0`). Helpers: `validate_drug`,
`get_overrides`, `apply_drug` (returns snapshot for restore), `restore_cfg`.

## Pending

- **Phase 3** — two-sim runner in `models/.../src/init.py` (per-condition
  loop: baseline + each drug; results merged into one `_data.pkl` with a
  new `drug_simData` key) + `models/.../src/batch.py` plumbing (validate
  `kwargs['drugs']`, set `DRUGS` env var, pass into `fitnessFuncArgs`).
- **Phase 4** — `RBS_network_models/fitnessFunc_v2.py`: detect
  `simData_pkl['drug_simData']`, compute simulated ratios per drug,
  compare to `<reference_h5>/drug_effects/<drug>/`, combine with baseline
  fitness via schema weights.
- **Phase 5** — `models/.../fitness_schema/schema_v3_drug.py`: mirror
  `schema_v3.py` and add a `drug_response` top-level component.
- **Phase 6** — `models/.../run_batch.py`: `--drugs` argparse flag,
  schema swap, drug-name validation against `DRUG_REGISTRY`.

## Bicuculline ratios — sanity reference

(For checking that the optimizer learns toward something biologically sensible.)

| Ratio | Value | Note |
|---|---:|---|
| `pop_FR_ratio` | 1.15 | total firing barely changed |
| `exc_firing_rate_ratio` | 1.99 | E ~doubles ✓ disinhibition |
| `inh_firing_rate_ratio` | 0.31 | top-20% baseline channels collapse |
| `network_burst_rate_ratio` | 2.15 | bursts double |
| `network_burst_duration_ratio` | 0.52 | bursts ~half as long |
| `burstlet_rate_ratio` | 1.52 | burstlets +52% |
| `burstlet_duration_ratio` | 0.73 | burstlets shorter |
| `pre_burstlet_rate_ratio` | 1.39 | permissive-threshold burstlets +39% |
| `pre_burstlet_duration_ratio` | 0.74 | also shorter |
| `superburst_rate_ratio` | 0.00 | superbursts wiped out |
| `superburst_duration_ratio` | 0.00 | (no superbursts post) |
| `mean_participation_ratio` | 0.18 | participation 62 % → 11 % |
