# Aim 2 — Approach Points (for grant draft)

> Notes-only document. Each top-level bullet is a self-contained point I can later convert into one paragraph or one passage in the Approach section. Sub-bullets are the supporting detail (technical specifics, justifications, alternatives, fallbacks). Order within each Aim follows the order I expect to use in the narrative.

---

## 0. Framing of the Aim 2 Approach (one or two paragraphs at the top)

- Aim 2 inherits a phenotypic atlas from Aim 1 and converts it into mechanism. The atlas tells us *what* differs between WT and each disorder; Aim 2 tells us *which biophysical processes are sufficient to produce that difference*, and *which of those processes are reversible* by pharmacology that can plausibly be moved into a clinical pipeline.
- The central methodological premise: a chip-scale biophysical model that has been *fit to* WT and disorder recordings is the only object on which we can perform the counterfactual experiments needed to separate primary cause from compensation. We cannot do this on the dish (we cannot subtract a single channel conductance from a neuron and remeasure the network). We *can* do it on the twin.
- The Aim is organized as a three-step loop: build twins of WT and disorder (2A), constrain those twins so they respond to drugs the way real cultures do (2B, anchoring), and refine the twins to single-unit resolution so cell-type-specific contributions can be read out (2C).
- Aim 2 does not require Aim 1 to be complete to begin. Preliminary HD-MEA datasets for KCNT1 GoF, Cdkl5, and folate-exposed cultures are already in hand. Aim 2A is therefore a "month-zero" aim. Aim 2C runs in parallel with 2A on the same recordings using a separate optimization track.
- Risk posture: 2A's success is assessed on whether the twin reproduces network-level statistics within a defined fitness window (see "Twin acceptance criteria" below). Even if a twin fails to fit a particular condition, the *failure mode* is informative — it identifies which biophysical degrees of freedom in our current parameter space are insufficient and points to the next experiment.

---

## 1. Aim 2A — Population-level twins and bidirectional sensitivity analysis

### 1.1 Inputs from Aim 1
- HD-MEA recordings (Maxwell, ~26k electrodes, 20 kHz) at DIV 14, 21, 28 for each of: WT, Kcnt1 GoF, Cdkl5 mutant, folic-acid-exposed cultures, with littermate controls in matched sessions.
- Per recording, two derived products are required as twin-fitting targets:
  - V1 (population) feature vector: mean firing rate, mean burst rate, mean burst duration, intra-burst spike rate, inter-burst interval, network burst rate/duration/participation, CV ISI, fano factor of population spike count, pairwise spike-time correlation (mean and distribution), rise/fall asymmetry of burst envelope.
  - V2 (sorted) feature vector: number of putative excitatory units, number of putative inhibitory units (HIPPIE waveform classifier), per-unit firing rate distribution, per-unit burstiness, mean cross-correlogram peak height by E/I pair type.
- The same feature extraction code that ran on the recording must run on the simulated chip — single source of truth, no separate "experimental" vs "simulated" pipeline. (Use `MEA_Analysis.NetworkAnalysis.awNetworkAnalysis.network_analysis.compute_network_metrics` for both, as already wired in `RBS_network_models.fitnessFunc`.)

### 1.2 Network model specification
- NetPyNE / NEURON biophysical network, scaled to the active recording footprint.
  - Population size: ~600–1000 neurons per twin; matched to the number of sorted units per recording when available, otherwise scaled to estimated active electrode count × yield factor.
  - E:I ratio default 80:20, but **E:I ratio is itself a free parameter** (see 1.3) because cell-type composition is one of the four candidate disrupted axes named in the Specific Aims.
  - Cell models: Hodgkin–Huxley with explicit Na, K-DR, K-A, KCa, L-type Ca, leak; soma + simplified dendritic compartment. Excitatory and inhibitory cells use separate channel-density priors centered on cortical pyramidal vs fast-spiking interneuron literature values.
  - Synapses: AMPA, NMDA, GABA-A, GABA-B with separate time constants and reversal potentials. NMDA voltage-dependent block included.
  - Connectivity: distance-dependent probability per E→E, E→I, I→E, I→I block; convergence and divergence allowed to vary independently.
- The model parameterization is the existing `evol_params` framework. The active variant on the `multi-drug-optimization` branch is the starting point and will be extended (see 1.3).

### 1.3 Parameter space — the four "axes" the Aim is built to decompose
The Specific Aims commit us to attributing each phenotype to a combination of **(a) intrinsic excitability, (b) synaptic strength/kinetics, (c) connectivity structure, (d) cell-type composition / homeostatic state**. The parameter vector must therefore have explicit, separable degrees of freedom along each axis. Concretely:
- (a) Intrinsic — per cell-type: gNa, gK-DR, gK-A, gKCa, gCaL, gleak, Cm, axial resistance. ~8 params × 2 cell types = ~16.
- (b) Synaptic — per connection class (4 classes): peak conductance, rise τ, decay τ, reversal (fixed), short-term plasticity facilitation/depression (optional). ~4–6 params × 4 classes = ~16–24.
- (c) Connectivity — per connection class: connection probability, distance length-constant, divergence cap. ~3 × 4 = 12.
- (d) Composition / homeostasis — global E:I ratio, global excitatory drive (background Poisson rate), homeostatic scaling factor on E synapses, homeostatic scaling factor on I synapses. ~4.
- Total ~50 free parameters. This is consistent with prior batches on this codebase and tractable inside NetPyNE Batch + inspyred at 512 candidates × ~100 generations on Perlmutter.
- **Critical design choice:** the four axes are kept dimensionally separate so that the sensitivity analysis in 1.6 can ask "how much of the WT→disorder shift lives on axis (a) vs (b) vs (c) vs (d)?" without the answer being an artifact of parameter coupling. Coupled parameters (e.g., gKCa entering both intrinsic excitability and burst termination) are flagged and analyzed jointly.

### 1.4 Fitness function
- A custom feature-based fitness function compares the simulated network's summary features against the experimental target h5 (one feature, one weight, one acceptable range per entry; structure exemplified in `fitness_schema/schema_2.py`). Per-condition variants of this schema weight the disease-discriminating features more heavily; we re-use the same schema machinery rather than introducing a new fitness formalism.
- Hard-rejection rules built into the schema (all-E or all-I firing, isolated neurons, runaway/silent firing, simulation crash) return the worst-case fitness so the optimizer prunes degenerate parameter regions automatically.

### 1.5 Optimization protocol — Optuna on a single Perlmutter CPU node
- The optimizer is Optuna (TPE sampler) running on one Perlmutter CPU node, with candidates evaluated in parallel across the node's CPU cores. Each well/condition/stage gets its own batch; this single-node-per-twin layout is what the existing `config_and_run_sbatch.sh` script (4.5 h, `-N 1`, `-C cpu`, `-A m2043`, image `balewski/ubu20-neuron8:v5`) already implements. No multi-node MPI fan-out is required for the optimizer itself; each candidate is one MPI-NEURON simulation inside the node, and Optuna distributes candidates across cores. This keeps queue wait short and lets us run many wells/conditions in parallel as independent jobs.
- The user-side workflow per recording is two scripts and nothing else:
  1. `_scripts/create_experimental_target.py` — reads spike-sorted `spike_times.npy` + `metrics_curated.xlsx` + `network_results.json` from MEA_Analysis output, classifies units (top-20% firing rate as putative inhibitory, rest excitatory), de-duplicates units sharing electrode locations (highest-FR kept), and writes the experimental target `.h5` under `processed_experimental_targets/`.
  2. `models/<MODEL>/<COND>/config_and_run.sh` — sets the cray-mpich + NEURON env, activates `preshifter`, and launches `run_batch.py <run_name> <evol_params_module>`, which is the Optuna driver that fits to the target h5 produced in step 1.
- **Dynamic time-window selection (built into step 1).** A standing problem with feature-based fitting on MEA recordings is that the experimental and simulated time windows must be chosen to make rate features comparable. We solve this analytically inside `create_experimental_target.py`: it sweeps a grid of candidate window lengths T (default 10–240 s), recomputes the rate features at each T, and picks the smallest T whose feature vector is closest (in schema-weighted absolute deviation against the active fit_schema) to the full-recording feature vector. The chosen T is written into the target h5 (`network_results.attrs['T_target_s']`) and is the simulation duration that `run_batch.py` reads back. This procedure makes the comparison window data-driven per recording rather than a hand-tuned constant — important because culture-to-culture and genotype-to-genotype activity densities differ enough that a fixed T systematically biases the fit.
- The simulated network is run for the same `T_target_s` so the fitness comparison is feature-for-feature, well-by-well. Spike trains are clipped at `T_target_s` on both sides, and per-unit `firing_rate` and `num_spikes` are recomputed from the clipped trains.
- Convergence criterion: per-feature errors on the top trial within 1.5× the cross-well biological variance for that feature, sustained over the last fraction of trials.
- Output structure already standardized: `z_simulated_data/<run>/batch_runs/batch_<date>_<tag>/gen_<n>/...` with per-trial cfg.json + data.pkl + plots, plus `fitness_summary/`. No new infrastructure needed.
- **Reproducibility:** each twin is reproduced with a second Optuna seed and (where relevant) a second fit_schema variant; a twin is only accepted if both replicates converge on overlapping regions of parameter space.

### 1.6 Bidirectional sensitivity analysis — extending the existing analysis pipeline
The sensitivity analysis we already use on completed Optuna runs is the foundation for this step. Optuna stores every evaluated parameter vector and its fitness, and we already extract per-parameter importance and parameter-vs-fitness response curves from those studies. The Aim 2 contribution is to apply that same machinery in a *bidirectional* way across paired WT and disorder studies, rather than introducing a new analytical framework:

- **Forward (WT → disorder).** Starting from the WT twin's top trials, we ask which parameter shifts are sufficient to move the simulated feature vector onto the disorder feature vector. In practice this is run as an Optuna study whose objective is the disorder feature vector but whose search is initialized at the WT optimum, with a sparsity penalty so the optimizer prefers solutions that change few parameters. The output is the parameter set whose perturbation alone reproduces the disorder phenotype — the candidate primary mechanism.
- **Reverse (disorder → WT).** Same procedure with the roles flipped: start at the disorder twin's optimum, target the WT feature vector, identify the smallest correction. The output is the candidate set of reversible targets.
- **Per-parameter importance.** For each direction, the existing Optuna importance analysis ranks which parameters carry the most fitness change; we group those parameters by the four biophysical axes from §1.3 (intrinsic / synaptic / connectivity / composition) so the result reads out as axis-level attribution rather than as a flat list of parameter names.
- **Why both directions are needed.** If forward and reverse agree on the same axes, the disorder is well-described as a one-way push along those axes and the reversible target coincides with the primary mechanism. If they disagree, the disorder includes a homeostatic compensation: forward reveals the primary perturbation, reverse reveals the more tractable therapeutic handle. This is exactly the primary-vs-compensation distinction the Aim is built to make.
- **Per-condition output.** For each disorder, an axis-level attribution plot (% of the WT→disorder shift living on each axis) computed separately for the forward and reverse directions, with confidence intervals derived from the top-trial cloud of the corresponding Optuna study.

### 1.7 Twin acceptance criteria (gate before passing to 2B)
A twin is "accepted" and may proceed to 2B only if:
- (i) Top candidate's per-feature errors are within 1.5× biological replicate variance.
- (ii) Two independent batches (different seeds, optionally different schemas) yield overlapping top-32 clouds.
- (iii) Held-out features (features computed but not included in fitness) are also reproduced within 2× variance — i.e., the twin generalizes beyond the targets it was fit to. **This is the single most important guard against overfitting.**
- (iv) The twin is not at a parameter-space boundary (boundary hits indicate the prior was wrong, not that the fit is good).

### 1.8 Risks and alternatives for 2A
- *Risk:* 50-dim parameter space is non-identifiable. *Mitigation:* the held-out feature criterion in §1.7 directly penalizes non-identifiability; if held-out features fail, we contract the parameter space to the most informative subset using mutual-information ranking on a screening batch.
- *Risk:* Disorder phenotype cannot be reached from WT prior. *Mitigation:* the disease-specific schema in §1.4 includes the "phenotype direction" dealbreaker; if the optimizer cannot reach the disorder regime, we widen the prior on the axis suggested by univariate fitting.
- *Risk:* Stochasticity of the simulator inflates fitness variance. *Mitigation:* every candidate is simulated with N≥3 random seeds and the fitness is the mean; this is already supported by the batch infrastructure.

---

## 2. Aim 2B — Pharmacological anchoring and validation

### 2.1 The problem 2B solves — the degeneracy problem
- A twin that matches baseline statistics may still be wrong about *which* biophysical knobs underlie those statistics. This is the well-known **degeneracy problem** in biophysical network fitting: many distinct parameter combinations yield the same population-level summary features (firing rates, burst metrics, synchrony). Optuna's top-trial cloud routinely contains parameter sets that are widely separated in parameter space yet near-identical in fitness, and any single one of them, taken on its own, would give a misleading mechanistic story.
- Without an independent constraint, 2A cannot tell us which member of the degenerate family is the *biophysically correct* twin. 2B supplies that constraint by requiring the twin to reproduce empirically observed pharmacological perturbations as well as baseline. A mechanism-specific drug (e.g., gabazine → GABA-A only) is effectively a constraint on the GABA-A axis: only the subset of degenerate baseline-fits that *also* respond correctly to the drug survive. Stacking enough such constraints collapses the degenerate family down to the parameter region whose mechanistic interpretation is biologically faithful — not just statistically faithful.
- A direct consequence — and an important framing point for the narrative — is that **not all twins from 2A will pass 2B.** Most candidates in the 2A top-trial cloud will fail one or more of the drug-response tests. That is the intended outcome: 2B is a filter that narrows 2A's broad solution set to the more constrained, more biophysically grounded subset, and the *fraction* of 2A solutions surviving is itself a quantitative measure of how identifiable each condition's parameter space is from baseline alone.

### 2.2 The decoder panel (mechanism-anchoring drugs, applied to WT and disorder cultures)
The decoder panel is anchored on the drugs for which we *already have paired pre/post HD-MEA recordings in hand* — these are the immediately tractable constraints and define the v1 panel. Other compounds are listed below at lower priority for future expansion.

- **Primary panel — already in hand, applied first:**
  - **AP5 + NBQX** — combined NMDA + AMPA glutamate block. In silico: `scale_NMDA = 0`, `scale_AMPA = 0`. The strongest single constraint: silences excitatory synaptic transmission, leaving only intrinsic and inhibitory dynamics, which makes it a sharp test of how much of the phenotype lives on the synaptic-excitatory axis.
  - **AP5** alone — NMDA receptor block. In silico: `scale_NMDA = 0`. Isolates NMDA's contribution to burst maintenance and plateau dynamics.
  - **Bicuculline** — GABA-A antagonist. In silico: `scale_GABA = 0`.
  - **Gabazine** — GABA-A antagonist (cleaner pharmacology than bicuculline, same in-silico mapping `scale_GABA = 0`); used as a confirmatory orthogonal probe of the GABA-A constraint.
  - **4-AP** — Kv channel block. In silico: `scale_K = 0`. Probes the intrinsic-K-current axis, the only intrinsic-side perturbation in the primary panel.
- **Secondary panel — lower priority, applied opportunistically as recordings become available:**
  - Diazepam (GABA-A potentiation), low-dose TTX (partial Na), apamin (SK / KCa), nimodipine (L-type Ca). Useful for resolving finer intrinsic-axis identifiability questions but not required for the v1 anchoring step.
- Each primary-panel drug is applied to N≥3 wells per genotype per developmental stage on which a baseline twin already exists.

### 2.3 Twin anchoring procedure — the multi-drug optimization workflow
The full implementation is laid out in `~/.claude/plans/so-in-this-repository-linked-cake.md` (multi-drug-optimization branch). The pipeline avoids the unit-matching problem (post-drug spike-sorting is unreliable because units shift) by working with **population-level rate-feature ratios** (post / pre) on both the experimental and the simulated side. Both sides are dimensionless and population-level, so the comparison is well-posed even though we cannot match individual neurons across conditions.

- **Experimental side — drug-effect ratio extraction.**
  - From paired pre/post raw MEA recordings, extract per-feature ratios over the same `T_target_s` window already chosen by the dynamic-window step in §1.5: pre is the spike-sorted baseline; post is MUA threshold-crossings on the same extremum channels (since post-drug sorting is unreliable). The ratio features are population firing rate, network-burst rate, burstlet rate, superburst rate, with mean burst duration and mean participation as secondary diagnostics.
  - These ratios are written into a new `/drug_effects/<drug_name>/` group inside the existing baseline target h5 (the same h5 that drives 2A), alongside a JSON-encoded record of which `cfg.scale_*` parameters define this drug's in-silico mapping, and the raw pre/post diagnostics that produced the ratios.
- **Simulation side — two-sim per candidate.**
  - Each Optuna candidate runs **all** registered drugs every trial (no sparse sampling — keeps the TPE objective stationary). A candidate trial therefore consists of a baseline simulation followed by one drug-perturbed simulation per registered drug, with the perturbation applied as a hand-coded `cfg.scale_*` override read from the drug registry (`scale_AMPA`, `scale_NMDA`, `scale_GABA`, `scale_K`, etc.). Drug intensities are *not* in the search space.
  - The same population-level rate features are computed for each simulated condition, simulated ratios are formed (drug / baseline), and a new `drug_response` schema component contributes the per-feature absolute deviation between simulated and experimental ratios. The total fitness combines baseline fitness and drug-response fitness with schema-defined weights.
- **Failure isolation.** The existing crash-safety net is extended so that any failed condition (baseline or any drug pass) sentinel'd to worst-case fitness covers all conditions for that candidate, preserving the "no MPI_Abort, no waiting" property the optimizer needs to keep moving.
- **What anchoring does to the parameter space — the constraining step.** The 2B fitness is *strictly more constraining* than 2A's: it is 2A's fitness plus a non-negative drug-response penalty. Every candidate that 2A would have accepted is still feasible, but the additional penalty re-ranks them, and the new top of the Optuna study is drawn from the subset that simultaneously matches baseline *and* matches every drug-response ratio. This is the operational sense in which 2B "fits a more constrained, more biophysical model": the same parameter space, the same simulator, but a fitness landscape with the degenerate baseline-equivalent solutions broken apart by the drug-response axis. The top-trial cloud after 2B is therefore the biophysically grounded subset of the 2A cloud, and the *shrinkage factor* — the fraction of the 2A cloud surviving 2B — is itself a quantitative readout of how degenerate baseline fitting was for that condition.

### 2.4 Disease-relevant validation drugs (test of reversibility predictions from 2A)
After anchoring, the twin's "reversible target" prediction from 2A is tested against pharmacology that targets exactly that axis.
- KCNT1 GoF: quinidine and bepridil (KNa channel blockers); compare predicted vs observed normalization.
  - In-silico equivalent requires adding KNa to the channel set if absent in the baseline model — a planned model extension specifically required for the KCNT1 condition.
- Cdkl5: ganaxolone (GABA-A allosteric PAM). Predicted to rescue if 2A reverse analysis points at the inhibitory-synaptic axis.
- Folate-exposed: BDNF supplementation and a psychoplastogen panel (TBG, low-dose ketamine). Predicted to rescue if 2A reverse analysis points at connectivity / synaptic-strength axes (the hypoconnectivity prediction).
- Each disease-relevant drug is applied at ≥3 concentrations per N≥3 wells per condition.

### 2.5 Decision logic — what does 2B let us conclude?
For each disorder × drug pairing, four possible outcomes:
1. *Twin predicted rescue, bench rescue observed* — primary mechanism is correctly identified; reversible target validated; pursue in Aim 3.
2. *Twin predicted rescue, bench rescue absent* — primary mechanism mis-specified, OR the twin is reading out a compensatory axis as if it were primary; refit with the drug response as additional fitness constraint and re-derive predictions.
3. *Twin predicted no effect, bench rescue observed* — there is a mechanism the twin does not encode; expand parameter space along that axis.
4. *Twin predicted no effect, bench no effect* — consistent with model.
- Each pairing's outcome is a distinct entry in the final mechanistic table delivered by Aim 2.

### 2.6 Risks and alternatives for 2B
- *Risk:* drug effects in dish are confounded by acute toxicity at the concentrations used. *Mitigation:* viability and waveform stability are checked across the recording; concentrations are titrated to the lowest dose producing a reliable effect.
- *Risk:* the in-silico drug analog (e.g., "scale gNa × 0.5") under-models the real drug. *Mitigation:* concentration–response is measured for each drug at three concentrations, and the twin must match the *shape* of the dose response, not just one concentration.
- *Risk:* not all decoder drugs are reversibly washable on HD-MEA. *Mitigation:* irreversible drugs are applied last in the protocol; fresh wells used for irreversible compounds.

---

## 3. Aim 2C — Single-unit-resolved twins

### 3.1 Why a unit-resolved twin is necessary in addition to the population twin in 2A
- **The biggest limitation of 2A is shared per-cell-type parameters.** In 2A every excitatory neuron in the simulated network is given the same intrinsic parameter set, and likewise every inhibitory neuron — there is one "E cell" and one "I cell," replicated. A direct consequence is that 2A twins almost always produce networks that fire *too regularly*: bursts onset, propagate, and terminate with much tighter timing than what we observe on the chip, and the per-unit firing-rate distribution is too narrow. Real cortical recordings show substantial unit-to-unit irregularity — heavy-tailed firing-rate distributions, jitter in burst participation, units that participate in some bursts and not others — and a homogeneous-cell-type model cannot reproduce that, no matter how well the population-summary features are fit. This irregularity gap is the single most important shortcoming 2C is built to close.
- Beyond the irregularity gap, the homogeneous-cell-type assumption also cannot localize a deficit to a *subtype* (e.g., PV vs SST inhibition; deep- vs superficial-layer pyramidal). HIPPIE waveform classification + electrical-footprint axonal tracking from Aim 1 give us subtype labels per unit; a unit-resolved twin uses those labels to ask "is the Kcnt1 phenotype carried by the fast-spiking subset specifically?" — a question 2A cannot answer.
- This is also where the framework becomes most translatable: in human iPSC co-cultures (Aim 3), the same per-unit fitting protocol applies and gives cell-population-resolved drug attribution.

### 3.2 Inputs from Aim 1 (V2 pipeline)
- Spike-sorted units with E/I subtype labels (HIPPIE).
- Per-unit waveform on the electrode array (footprint).
- Per-unit axonal skeleton from electrical-footprint propagation.
- Pairwise cross-correlograms between every unit pair, with the short-latency monosynaptic peak fit to a template to estimate sign and putative weight (Spivak/English-style).

### 3.3 Two-stage optimization (per-unit local fit, then global refinement)
- **Stage 1 — per-unit local fit (parallel, embarrassingly so):**
  - For each sorted unit, fit a single-cell biophysical model (HH, single soma + 1–2 dendritic compartments) to reproduce that unit's:
    - Mean firing rate.
    - ISI distribution shape (CV, skew).
    - Burst participation profile (probability of firing in each network-burst window).
    - First-spike latency in each network burst.
  - Inputs to this single-cell model: a synthetic synaptic drive constructed from the unit's incoming CCG-derived weights × the source units' actual spike trains (i.e., the rest of the network is *replayed* from the recording, not simulated).
  - Output: a per-unit intrinsic parameter vector (8–12 params) and a CCG-derived incoming-weight vector (≤ N_presynaptic).
  - This stage is parallel across units and across wells; cost scales linearly. With 500 units × ~100 candidates × 50 generations × 30 s/sim, well within Perlmutter budget for one batch per genotype per developmental stage.
- **Stage 2 — global wiring refinement:**
  - Assemble the full network with the Stage 1 per-unit intrinsics fixed and the CCG-derived weights as priors.
  - Allow connection weights to be re-tuned (within priors) so that the fully simulated network — with no recorded-spike replay, only the simulated network firing itself — reproduces the population-level feature vector from 2A.
  - This stage is a smaller-dimensional fit than 2A (intrinsics already fixed) but operates on a much larger network instance.

### 3.4 Subtype-specific bidirectional sensitivity analysis
- Repeat the forward and reverse analyses from §1.6, now restricting Δθ to specific subtype groups:
  - "What if only fast-spiking inhibitory unit intrinsics shift?"
  - "What if only deep-layer pyramidal connectivity shifts?"
- Each restricted Δθ gives a subtype-attribution score for the phenotype.
- Final per-condition output: a subtype × axis matrix showing which subtype × which biophysical axis carries each phenotype.

### 3.5 Cross-validation between 2A and 2C
- The unit-resolved twin's predicted population statistics must match the population-twin's predicted population statistics within tolerance. If they disagree, the disagreement is logged as a model-selection finding (the population-level abstraction was missing a degree of freedom revealed by per-unit fitting).
- Conversely, the unit-resolved twin must reproduce the decoder-drug responses from 2B at unit resolution — i.e., the unit-by-unit change in firing rate under, e.g., NBQX, should match.

### 3.6 Risks and alternatives for 2C
- *Risk:* CCG-derived weights are biased estimators of true synaptic weight. *Mitigation:* CCG weights are used as *priors* with broad variance, not fixed values, and Stage 2 retunes them; we also compare to GLM-based estimators on a subset.
- *Risk:* Stage 1 per-unit fitting overfits to replayed spikes. *Mitigation:* held-out time windows; per-unit fits validated on segments not used in optimization.
- *Risk:* HIPPIE subtype calls are noisy on cultured (vs in-vivo) neurons. *Mitigation:* sensitivity analyses restricted by subtype are repeated under shuffled subtype labels as a null; real subtype-attribution scores are reported only if they exceed the shuffle distribution.
- *Risk:* The number of free parameters scales with the number of units, raising overfitting concerns. *Mitigation:* per-unit intrinsics are regularized toward subtype-mean priors (hierarchical model); deviations are penalized unless the unit's data demand them.

---

## 4. Computational infrastructure (covers all of 2A/2B/2C)

- Existing pipeline on this codebase already supports 2A end-to-end; the multi-drug-optimization branch is the working substrate. No new framework is required for 2A.
- 2B requires a multi-condition fitness evaluator: a single candidate is simulated under baseline + each in-silico drug, and the fitness aggregates across all conditions. Implementation is an extension of the existing fitness function (one extra dimension in the metric vector); preliminary work for this is on the multi-drug-optimization branch.
- 2C requires (i) a per-unit replay simulator (single-cell sim driven by recorded spikes) and (ii) a coupling between Stage 1 per-unit fits and Stage 2 global refinement. The Stage 1 simulator is a new module; Stage 2 reuses existing NetPyNE Batch infrastructure.
- Compute footprint estimate (per condition × per developmental stage):
  - 2A baseline twin: ~250 node-hours.
  - 2B anchored twin: ~600 node-hours (multi-condition fitness ~3× cost).
  - 2C unit-resolved twin: ~400 node-hours (Stage 1) + ~250 node-hours (Stage 2).
  - Total ~1500 node-hours per condition per stage × 4 conditions × 3 stages ≈ 18,000 node-hours over the project. Within typical NERSC m2043 allocation with renewal.
- All twins, fitness summaries, parameter clouds, and sensitivity-analysis outputs are versioned under `z_simulated_data/<run_name>/...` with a manifest tying each twin to (condition, stage, schema version, code commit hash, container tag).

---

## 5. Deliverables and milestones for Aim 2

- **Year 1, Q2:** WT twins accepted at all three developmental stages; 2A pipeline validated.
- **Year 1, Q4:** Disorder twins for all three conditions accepted at DIV 21 (the densest recording timepoint); first forward/reverse sensitivity analyses delivered.
- **Year 2, Q2:** Decoder-anchored twins for all three conditions, all stages; 2B prediction table for disease-relevant drugs locked.
- **Year 2, Q4:** Bench validation of disease-relevant drug predictions complete for all three conditions; outcome table per §2.5 finalized.
- **Year 3, Q2:** 2C unit-resolved twins delivered for KCNT1 and Cdkl5 (best-sorted conditions). Subtype × axis attribution tables produced.
- **Year 3, Q4:** Cross-validation between 2A and 2C complete; full mechanistic decomposition delivered to Aim 3 for translation to human systems.

---

## 6. What 2A/2B/2C deliver as mechanism (the one-line take-home for each condition)

For each of the three NDD models, Aim 2 produces a single integrated statement of the form:

> "Phenotype X in condition Y is attributable [primarily / jointly] to perturbation of axis A_i [and A_j], localized to subtype S [if 2C resolves it]; the perturbation is [reversible / non-reversible] in silico by correction of axis A_k, and pharmacological agent P targeting A_k [does / does not] normalize the phenotype on the bench."

That sentence — produced for each of Kcnt1, Cdkl5, and folate exposure — is the deliverable Aim 2 owes the rest of the proposal, and it is what Aim 3 carries into the human iPSC and organoid systems.
