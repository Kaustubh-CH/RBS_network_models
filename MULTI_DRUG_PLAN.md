# Multi-Drug Optimization

## Context

Today the optimizer fits a single condition: simulated baseline network → spike-sorted baseline experimental features. We want to additionally fit *how the network responds to pharmacological perturbations*. The biological signal is recorded as paired raw MEA recordings (baseline + post-drug, e.g., bicuculline blocks GABA-A). Spike-sorting the post-drug recording is unreliable because units shift, so we cannot match individual neurons across conditions.

**Strategy**: extract population-level rate-feature *ratios* (post / pre) from the raw recordings, store them as the experimental drug-response signature, and during optimization run each candidate twice (baseline + drug-perturbed). The drug perturbation is a hand-coded parameter override (e.g., `cfg.scale_GABA=0` for bicuculline). Fitness compares **simulated ratios vs experimental ratios** — both sides dimensionless and pop-level, sidestepping the unit-matching problem.

**User-confirmed design choices**:
- Hand-coded drug → cfg-param mappings (no drug intensity in the optimizer search space).
- Each candidate evaluates **all** registered drugs every generation (N=1, no sparse sampling — keeps Optuna's TPE objective stationary).
- List-of-drugs plumbing from day 1 (`--drugs bicuculline gabazine`); v1 ships with a single drug registered.
- Ratios stored under a new `/drug_effects/<drug_name>/` group **inside the existing baseline target h5**.

**Existing infrastructure to reuse** (do not rebuild):
- `cfg.scale_AMPA / scale_NMDA / scale_GABA / scale_K` already defined at `models/.../src/cfg.py:116-119` and applied at `netParams.py:245-246, 265`. The cfg.py comment block at lines 110-115 even pre-specs the drug→param mappings — we codify those into a registry.
- Pre/post raw-recording MUA pipeline in `MEA_Analysis/IPNAnalysis/spikesort_drug_comparison.py` (extremum-channel carryover, identical preprocessing, `detect_threshold_crossings`) and existing outputs at `MEA_Analysis/results_drug_comparison/CDKL5_02_vs_Imm11/well*/pre_post_thresh_crossings.npz`.
- `parameter_free_burst_detector.compute_network_bursts` works on any `{unit_id: spike_times}` dict — applies cleanly to MUA threshold-crossings.
- Crash-safety net in `init.py:91-103` already turns sim failures into worst-fitness sentinels; we extend it to cover both baseline and drug passes.

## Branching

In `RBS_network_models/` (independent git repo): `git checkout -b multi-drug-optimization`. Implementation phases below all happen on that branch.

---

## Phase 1 — Experimental drug-effect ratio extraction

**New script**: `RBS_network_models/_scripts/extract_drug_effects.py`

```
extract_drug_effects.py \
  --baseline_target /.../processed_experimental_targets/CDKL5_002_well000.h5 \
  --baseline_raw /.../experimental_data/CDKL5_02.h5 \
  --drug_raw     /.../experimental_data/Immediately_after_drug_11.h5 \
  --drug_name    bicuculline \
  --baseline_analyzer /.../sorting_analyzer_dir   # for extremum channels
  [--T_seconds 90]                                # window for both recordings; default = baseline h5's T_target_s
  [--from_existing_results /.../results_drug_comparison/CDKL5_02_vs_Imm11/well000/]
```

Behavior:
1. Load baseline target h5; pull `T_target_s` from `network_results.attrs` and use it as the comparison window for both recordings (clip spikes/MUA to `[0, T]`).
2. **Pre-side spikes**: read `unit_*/spike_times` from baseline target h5 → dict `{uid: times}`.
3. **Post-side MUA**: reuse `spikesort_drug_comparison.preprocess_recording` + `detect_threshold_crossings` to extract threshold-crossing times per extremum channel (extremum channels carried over from `--baseline_analyzer`). Convert to `{channel_id: times}` dict — treat each extremum channel as a "unit" for `compute_network_bursts`.
4. **Run network burst detection** twice with the production params from `fitnessFunc_v2._run_burst_detector`:
   - `pre_nb = compute_network_bursts(SpikeTimes=pre_units, plot=False, base_threshold_static=40, min_burstlet_participation=0.05)`
   - `post_nb = compute_network_bursts(SpikeTimes=post_mua, plot=False, base_threshold_static=40, min_burstlet_participation=0.05)`
5. **Compute population firing rate**: `pop_FR = total_spikes / (n_units * T)` for each side.
6. **Compute ratios** (post / pre) for each rate feature with NaN-safe division (epsilon floor `1e-9`):
   - `pop_FR_ratio`
   - `network_burst_rate_ratio` (from `nb["network_bursts"]["metrics"]["rate"]`)
   - `burstlet_rate_ratio`
   - `superburst_rate_ratio`
   - Optional secondary: `mean_burst_duration_ratio`, `mean_participation_ratio`.
7. **Write into baseline h5** under new group `/drug_effects/<drug_name>/`:
   - attrs: `drug_name`, `drug_param_overrides` (JSON-encoded, e.g. `{"scale_GABA": 0.0}` from the registry), `baseline_raw_path`, `drug_raw_path`, `T_seconds`, `extraction_method`, `extraction_timestamp`.
   - datasets: one per ratio feature (scalar). Plus diagnostics: `pre_pop_FR_hz`, `post_pop_FR_hz`, `pre_nb_rate_hz`, `post_nb_rate_hz`.
8. Idempotent: if `/drug_effects/<drug_name>/` exists, overwrite (with a `--no_overwrite` guard).

Helper `--from_existing_results` mode: read `pre_post_thresh_crossings.npz` directly to skip steps 2-3 — most usage will go through this path since the comparison results already exist.

---

## Phase 2 — Drug perturbation registry

**New module**: `RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/src/drug_perturbations.py`

```python
DRUG_REGISTRY = {
    "bicuculline": {
        "cfg_overrides": {"scale_GABA": 0.0},
        "description": "GABA-A antagonist; full block",
    },
    "gabazine": {
        "cfg_overrides": {"scale_GABA": 0.0},
        "description": "GABA-A antagonist; full block",
    },
    "ap5_nbqx": {
        "cfg_overrides": {"scale_AMPA": 0.0, "scale_NMDA": 0.0},
        "description": "AMPA + NMDA glutamate block",
    },
    "ap5": {
        "cfg_overrides": {"scale_NMDA": 0.0},
        "description": "NMDA block",
    },
    "fourap": {
        "cfg_overrides": {"scale_K": 0.0},
        "description": "Kv channel block",
    },
}

def apply_drug(cfg, drug_name: str) -> dict:
    """Mutate cfg in-place; return the dict of {param: (prev, new)} for logging/restore."""
```

Single source of truth — both `init.py` (to perturb the live cfg) and `extract_drug_effects.py` (to record `drug_param_overrides` into the h5) read from this registry. Mirrors the comment at `cfg.py:110-115`.

---

## Phase 3 — Two-sim init.py + batch.py plumbing

### init.py — list-shaped multi-condition runner

**File**: `models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/src/init.py`

Replace the single-sim body (currently lines 91-103) with a per-condition loop:

```python
conditions = [("baseline", {})]
drugs_env = os.environ.get("DRUGS", "").strip()
if drugs_env:
    for drug in drugs_env.split(","):
        conditions.append((drug, DRUG_REGISTRY[drug]["cfg_overrides"]))

results = {}
for cond_name, overrides in conditions:
    snapshot = {k: getattr(simConfig, k) for k in overrides}
    for k, v in overrides.items():
        setattr(simConfig, k, v)
    try:
        sim.create(simConfig=simConfig, netParams=netParams)
        sim.simulate()
        sim.analyze()
        results[cond_name] = {
            "spkt": list(sim.simData["spkt"]),
            "spkid": list(sim.simData["spkid"]),
        }
    except BaseException as e:
        results[cond_name] = {"_failure": True, "_failure_reason": f"{type(e).__name__}: {e}"}
    finally:
        for k, v in snapshot.items():
            setattr(simConfig, k, v)
```

After all conditions run, rank 0 writes a single `<simLabel>_data.pkl` with shape:
```
{
  "simData": { ...baseline spkt/spkid + V_soma + t... },     # back-compat: top-level still baseline
  "simConfig": ..., "netParams": ..., "netCells": ..., "netPops": ...,
  "drug_simData": {                                           # NEW
      "bicuculline": {"spkt": [...], "spkid": [...]},
      ...
  },
  "_failure": True/False, "_failure_reason": ... (only if any condition failed)
}
```

Existing crash-safety net (`_write_sentinel` at `init.py:58-89`) extends to: if **any** condition fails, write a sentinel covering all drugs (`drug_simData={drug: {"_failure": True}}`) plus a worst-fitness `_fitness.json`. This preserves the "no MPI_Abort, no waiting" property.

### batch.py — env var plumbing

**File**: `models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/src/batch.py`

- Read `kwargs.get('drugs', [])` — list of drug names.
- Validate each name against `DRUG_REGISTRY`; exit cleanly on typos.
- Set `os.environ['DRUGS'] = ','.join(drugs)` so srun-launched init.py sees it (mirrors the existing `T_TARGET_S` env-var pattern).
- Add `'drugs': drugs` to `fitnessFuncArgs` so the fitness function knows which drugs to score.

---

## Phase 4 — Fitness function: ratio-based drug response scoring

**File**: `RBS_network_models/RBS_network_models/fitnessFunc_v2.py`

Add a new `drug_response` component to the existing `fitnessFunc_v2` flow (existing flow described at `fitnessFunc_v2.py:1516-1665`):

1. After loading baseline simData (existing code unchanged), check for `simData_pkl["drug_simData"]`.
2. For each drug present:
   - Compute baseline-sim and drug-sim features over the same `T_target_s` window:
     - Population firing rate: `pop_FR = len(spkt) / (n_cells * T_seconds)`.
     - Network burst rate, burstlet rate, superburst rate via the existing `_run_burst_detector` helper (same `base_threshold_static=40, min_burstlet_participation=0.05` params already used for the baseline path).
   - Compute simulated ratios: `sim_ratio[f] = drug_sim[f] / max(baseline_sim[f], 1e-9)` for each feature `f`.
3. Read experimental ratios from `<reference_h5>/drug_effects/<drug>/`.
4. Per-feature loss = `|sim_ratio - exp_ratio|` weighted by `max_dev` from the schema (default to absolute deviation; relative deviation is a tunable knob if absolute is poorly calibrated).
5. Sum per-feature losses with the new `drug_response` schema component, mean across drugs (so adding more drugs doesn't blow up the scale).
6. Combined fitness: `f_total = w_baseline * f_baseline + w_drug * f_drug_response`. Both `w_baseline` and `w_drug` come from the schema's top-level component weights.

If `drug_simData` is absent (legacy run, or any drug-pass failed and was sentinel'd), `f_drug_response = max_score`. The optimizer learns to avoid those parameter regions.

---

## Phase 5 — Drug response schema

**New file**: `models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/fitness_schema/schema_v3_drug.py`

Mirror `schema_v3.py` and add a `drug_response` component:

```python
fit_schema = {
  "max_fitness": 30000.0,
  "components": {
    "unit_metrics":         {"weight": 0.30, "metrics": {...same as schema_v3...}},
    "hierarchical_bursts":  {"weight": 0.30, "metrics": {...same as schema_v3...}},
    "drug_response": {
      "weight": 0.40,
      "metrics": {
        "pop_FR_ratio":             {"weight": 0.30, "max_dev": 1.0},
        "network_burst_rate_ratio": {"weight": 0.40, "max_dev": 1.5},
        "burstlet_rate_ratio":      {"weight": 0.20, "max_dev": 1.5},
        "superburst_rate_ratio":    {"weight": 0.10, "max_dev": 2.0},
      },
    },
  },
}
```

`drug_response.weight = 0.40` is a starting point — re-tune from data after the first runs.

---

## Phase 6 — run_batch.py CLI integration

**File**: `models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/run_batch.py`

Add an argparse flag mirroring the existing `--T_target` pattern at lines 37-55:

```python
_parser.add_argument('--drugs', nargs='*', default=[],
                     help='Drug condition names (e.g., bicuculline). '
                          'Each candidate evaluates all listed drugs every generation.')
```

Plumb `drugs=_known.drugs` into `kwargs`. When `--drugs` is non-empty, switch the active fit_schema import to `schema_v3_drug`. Validate names against `DRUG_REGISTRY` early.

---

## Files added / modified

**ADD**:
- `RBS_network_models/_scripts/extract_drug_effects.py`
- `RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/src/drug_perturbations.py`
- `RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/fitness_schema/schema_v3_drug.py`

**MODIFY**:
- `RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/src/init.py` — list-shaped multi-condition sim runner; sentinel covers all conditions.
- `RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/src/batch.py` — read `kwargs['drugs']`, validate, set `DRUGS` env var, pass `drugs` into `fitnessFuncArgs`.
- `RBS_network_models/RBS_network_models/fitnessFunc_v2.py` — `drug_response` scoring component; reads `simData_pkl['drug_simData']` and `<reference_h5>/drug_effects/<drug>/`.
- `RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/run_batch.py` — `--drugs` CLI flag, schema swap.

**LEAVE ALONE**:
- `cfg.py`, `netParams.py` — `scale_*` hooks already present, no edits needed.
- `evol_params_from_seed_v3_large_v3.py` — drug intensities are not in the search space (per "hand-coded" choice).
- `MEA_Analysis/IPNAnalysis/spikesort_drug_*.py` — used as-is by `extract_drug_effects.py`.

---

## Verification

1. **Phase 1 standalone**:
   ```bash
   module load conda && conda activate preshifter
   python RBS_network_models/_scripts/extract_drug_effects.py \
     --baseline_target /pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/processed_experimental_targets/CDKL5_002_well000.h5 \
     --from_existing_results /pscratch/sd/k/ktub1999/networkSimulations/MEA_Analysis/results_drug_comparison/CDKL5_02_vs_Imm11/well000/ \
     --drug_name bicuculline
   ```
   Then `h5dump -A /pscratch/.../CDKL5_002_well000.h5 | grep -A2 drug_effects` to confirm the new group exists with all ratio datasets and the `drug_param_overrides` attr.

2. **End-to-end smoke (login node, single candidate, 1 drug, T=20)**:
   ```bash
   shifter --image=adammwea/netsims_docker:v1 /bin/bash
   # temporarily set maxiters=1, popsize=1 in run_batch.py
   python run_batch.py --drugs bicuculline --T_target 20 smoketest_multidrug
   ```
   Expect: trial dir's `_data.pkl` contains both `simData` and `drug_simData['bicuculline']` populated; `_fitness.json` shows non-zero `drug_response` component.

3. **Baseline-only regression**:
   ```bash
   python run_batch.py --T_target 20 baseline_only_regression
   ```
   No `--drugs` → identical to today. Diff `_data.pkl` against a pre-change baseline run to confirm no `drug_simData` key written and fitness numerically matches.

4. **Failure-isolation**:
   Force a drug failure (e.g., `scale_GABA=0` combined with a high-weight GABA seed known to drive runaway depolarisation) and confirm:
   - The candidate gets a max-fitness sentinel (not MPI_Abort).
   - Optuna proceeds to the next candidate without `maxiter_wait` timeout.

5. **Small batch on Perlmutter** (1 generation × 8 candidates × 1 drug, T=20, 2 nodes):
   ```bash
   sbatch --nodes=2 --ntasks-per-node=128 --time=01:00:00 \
     queue_job.sh ... --drugs bicuculline
   ```
   Confirm wall-clock is ≤2× the equivalent baseline-only run, and `fitness_summary/all_fitness_values.csv` includes the `drug_response` column.

6. **Optuna sanity**: after step 5, plot fitness vs generation — confirm the surrogate proposes monotonically improving candidates (no obvious sign that doubled per-trial cost broke convergence telemetry).

---

## Out of scope (future)

- Acquiring more drug-pair recordings (currently only one available: `CDKL5_02` ↔ `Immediately_after_drug_11`).
- Optimizer-learned drug intensities (would add `scale_GABA_drug` to the search space).
- E/I-resolved drug response — current ratios are population-level only because post-drug spike-sorting isn't trustworthy.
- Drug-onset *dynamics* (wash-in transients) — we model only steady-state pre/post.
- Drug effects on synaptic kinetics (`tau1_GABA`, `tau2_GABA`) — only conductance scaling for v1.
