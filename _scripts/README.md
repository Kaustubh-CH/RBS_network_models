# Trial rerun outputs

`rerun_trial_save_spikes_conn.py` re-runs a fitted trial from its `trial_*_cfg.json`
and writes four things per run, all prefixed with the sim label
(`<trial>_rerun_<duration>s` unless `--sim-label` is given):

| File | What it is |
|---|---|
| `*_data.pkl` | full NetPyNE output (written by NetPyNE itself) |

## What is in `*_data.pkl`

Five top-level keys: `netpyne_version`, `netpyne_changeset`, `net`, `simConfig`,
`simData`. Sizes below are from `op_gen49_trial49_1hr/trial_49_rerun_3600s_data.pkl`
(3605 s, 533 cells, 3.74 M spikes, 182 MB on disk).

### `simData` — 134.7 MB

| Key | Contents |
|---|---|
| `spkt` | spike times, **milliseconds**, Python list (3,739,594 entries) |
| `spkid` | matching cell gid per spike, stored as floats |
| `t` | **empty** — no time vector, see below |
| `popRates` | mean rate per population, e.g. `{'E': 0.257, 'I': 8.082}` Hz |
| `avgRate` | network mean rate, Hz |

### `net` — 55.5 MB

- **`cells`** (54.6 MB) — one entry per cell, each with `gid`, `tags`
  (`pop`, `cellType`, `cellLabel`, `x`/`y`/`z` and normalised coords), `conns`,
  `stims`, `secs`, `secLists`. Morphology is a single `soma` section holding
  `mechs`, `geom`, `topol`, `synMechs`.
  Every synaptic contact lives here on its **post**-synaptic cell, as
  `preGid`, `sec`, `loc`, `synMech`, `weight`, `delay`, `label`, `preLoc`
  (182,575 contacts for this network). `stims` is empty — no external drive.
- **`pops`** — `E` (gids 0–417) and `I` (gids 418–532), each with `tags` + `cellGids`.
  NetPyNE assigns gids in pop-creation order, E first, then I.
- **`params`** (0.6 MB) — the full netParams: `cellParams`, `connParams`,
  `popParams`, `synMechParams`, `subConnParams`, `stimSourceParams` /
  `stimTargetParams`, plus `propVelocity`, `scale`, `sizeX/Y/Z`, `shape` and the
  default weight / delay / threshold settings.

### `simConfig` — negligible size, 116 entries

The exact config that ran, including `duration` (ms), `duration_seconds`,
`network_cool_down`, `dt`, `cvode_active`, `propVelocity`, `num_excite`,
`num_inhib`, `simLabel`, `saveFolder`.
