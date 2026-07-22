import matplotlib; matplotlib.use('Agg')
import os
import sys
import json
import pickle
import traceback
from netpyne import sim
import numpy as np
#===================================================================================================
try:
    # Get and test MPI rank
    from mpi4py import MPI
    mpi_rank = MPI.COMM_WORLD.Get_rank()
    mpi_size = MPI.COMM_WORLD.Get_size()
    rank = mpi_rank
    print("Initiating Rank:", rank)
except ImportError:
    print("mpi4py not found, running in serial mode.")
    rank = 0
    mpi_size = 1
    MPI = None

#===================================================================================================

#set paths to simulation configuration files
simCfg = 'cfg.py'
netParams = 'netParams.py'

# initialize simConfig and netParams objects
simConfig, netParams = sim.readCmdLineArgs(
    simConfigDefault=simCfg,
    netParamsDefault=netParams,
)
# simConfig.verbose = True

# ===================================================================================================
# Crash-safety net
#
# A single diverging candidate (CVode "error test failed", runaway depolarisation,
# NaN in gating variables, etc.) used to MPI_Abort the entire srun, killing all
# 128 ranks of THIS trial AND making the optimizer wait `maxiter_wait * time_sleep`
# seconds before giving up. This wrapper turns any catchable failure into a
# worst-fitness sentinel on disk, so:
#   - the optimizer sees `<simLabel>_data.pkl` + `<simLabel>_fitness.json`
#     immediately and proceeds to the next candidate (no wait),
#   - the failed candidate's fitness is the schema's max_fitness, so optuna
#     learns to avoid that parameter basin instead of getting silent garbage.
#
# We can only catch failures NEURON exposes as Python exceptions. A truly
# C-level MPI_Abort still slips through; for those the optimizer still
# falls back to its existing wait-and-skip logic.
# ===================================================================================================

_save_folder = getattr(simConfig, 'saveFolder', '.')
_sim_label = getattr(simConfig, 'simLabel', 'trial')
_max_fitness = float(getattr(simConfig, 'max_fitness', 10000.0))

def _write_sentinel(reason: str):
    """Rank-0 writes a worst-fitness <simLabel>_data.pkl + <simLabel>_fitness.json
    so the orchestrator sees the candidate as immediately scored."""
    if rank != 0:
        return
    try:
        os.makedirs(_save_folder, exist_ok=True)
        try:
            simcfg_dict = simConfig.todict()
        except Exception:
            simcfg_dict = {}
        data = {
            'simData': {
                'spkt': [], 'spkid': [], 't': [],
                'V_soma': {}, 'avgRate': 0.0, 'avgRate_E': 0.0, 'avgRate_I': 0.0,
            },
            'simConfig': simcfg_dict,
            'netParams': {},
            'netCells': [],
            'netPops': {},
            '_failure': True,
            '_failure_reason': reason,
        }
        data_path = os.path.join(_save_folder, f"{_sim_label}_data.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
        fit_path = os.path.join(_save_folder, f"{_sim_label}_fitness.json")
        with open(fit_path, 'w') as f:
            json.dump({'fit': _max_fitness, 'failure': True, 'reason': reason}, f)
        print(f"[init.py] sentinel written for {_sim_label}: {reason[:200]}", flush=True)
    except Exception as e:
        print(f"[init.py] failed to write sentinel: {type(e).__name__}: {e}", flush=True)

# ===================================================================================================
# Segmented run with a coarse-integration cool-down
#
# The first `network_cool_down` seconds of every sim are a settling transient that
# fitnessFunc_v2 discards (extract_simulated_features drops spikes < cool_down and
# shifts the rest to t=0). Since that window is thrown away, we integrate it
# cheaply: this model uses cfg.cvode_active=True (adaptive timestep), so the knob
# equivalent to "bigger dt" is the CVode tolerance — we loosen cvode.atol by
# COOLDOWN_ATOL_SCALE for [0, cool_down], then restore the production atol for the
# scored window [cool_down, duration]. (If CVode is off we bump fixed h.dt instead.)
# This replaces NetPyNE's sim.runSim() (preRun + finitialize + psolve(duration));
# gatherData()/analyze() still run afterward exactly as before.
#
# Caveat: the network STATE handed to the scored window at t=cool_down is then
# approximate — fine for a settling period, but a very loose scale near a
# bifurcation could shift which basin the network settles into. Tune via the
# COOLDOWN_ATOL_SCALE env var (default 20; set 1 to disable coarsening).
# ===================================================================================================

_cool_down_ms = float(getattr(simConfig, 'network_cool_down', 0.0) or 0.0) * 1000.0
try:
    _cooldown_atol_scale = float(os.environ.get('COOLDOWN_ATOL_SCALE', '20'))
except ValueError:
    _cooldown_atol_scale = 20.0

def _run_sim_with_cooldown():
    """Drop-in for sim.runSim() that integrates the cool-down window with a
    coarser tolerance/timestep. Mirrors netpyne.sim.run.runSim + postRun."""
    from neuron import h
    sim.pc.barrier()
    if getattr(sim.cfg, 'use_local_dt', False):
        try:
            sim.cvode.use_local_dt(1)
        except Exception:
            pass
    sim.pc.barrier()
    sim.timing('start', 'runTime')
    sim.preRun()  # sets cvode.active/atol = cfg.cvode_atol, schedules recording events
    h.finitialize(float(sim.cfg.hParams['v_init']))

    _dur = float(sim.cfg.duration)
    _cd = min(_cool_down_ms, _dur)
    _coarsen = _cd > 0 and _cooldown_atol_scale and _cooldown_atol_scale > 1.0
    if _coarsen:
        if bool(int(getattr(sim.cfg, 'cvode_active', 0))):
            _fine = float(getattr(sim.cfg, 'cvode_atol', 1e-3))
            sim.cvode.atol(_fine * _cooldown_atol_scale)  # coarse: larger adaptive steps
            sim.pc.psolve(_cd)
            sim.cvode.atol(_fine)                          # restore production tolerance
        else:
            _base_dt = float(h.dt)
            h.dt = _base_dt * _cooldown_atol_scale         # coarse fixed timestep
            sim.pc.psolve(_cd)
            h.dt = _base_dt
        sim.pc.psolve(_dur)
    else:
        sim.pc.psolve(_dur)

    sim.pc.barrier()
    sim.timing('stop', 'runTime')
    if rank == 0:
        print('  Done; run time = %0.2f s (cool-down %0.1fs coarsened x%.0f)'
              % (sim.timingData['runTime'], _cd / 1000.0,
                 _cooldown_atol_scale if _coarsen else 1.0), flush=True)

# ===================================================================================================
# Per-condition simulation loop: baseline + (optionally) one sim per drug listed
# in os.environ['DRUGS'] (set by batch.py when run_batch.py --drugs ... is used).
#
# The baseline sim runs the decomposed createSimulateAnalyze (create -> run ->
# gather -> analyze, with the cool-down-coarsened run from _run_sim_with_cooldown)
# so the on-disk _data.pkl matches every existing consumer (plot_batch,
# rerun_trial_from_cfg, fitnessFunc_v2 baseline scoring). After baseline, each drug pass mutates
# simConfig's pharmacological scalers (scale_AMPA/NMDA/GABA/K) via the
# DRUG_REGISTRY in src/drug_perturbations.py, rebuilds the network, runs the
# sim, and captures only spike trains (spkt/spkid). Rank 0 then reloads the
# baseline pkl and merges drug_simData={drug: {spkt, spkid}, ...} into it so
# the per-drug data lands in the same single pkl per trial.
# ===================================================================================================

_drugs_env = os.environ.get("DRUGS", "").strip()
_drug_list = [d.strip() for d in _drugs_env.split(",") if d.strip()] if _drugs_env else []

# Import drug helpers only when needed so baseline-only runs stay isolated from
# any drug_perturbations.py syntax error.
if _drug_list:
    from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.src.drug_perturbations \
        import (apply_drug, restore_cfg, validate_drug,
                apply_drug_to_netparams, restore_netparams)
    for _d in _drug_list:
        validate_drug(_d)

# ---------------------------------------------------------------------------
# CRITICAL ORDERING CONSTRAINT — why the baseline pass saves to a temp label
#
# NetPyNE's mpi_direct orchestrator (netpyne/batch/utils.py) treats the mere
# *appearance* of `<simLabel>_data.pkl` as the trial-complete signal: the moment
# that file exists with a `simData` key it scores fitness, then kills every
# `nrniv` process in the generation. If the baseline pass saved straight to the
# final pkl path (as createSimulateAnalyze does by default), the orchestrator
# would score the candidate from a baseline-only pkl BEFORE any drug pass ran —
# so drug_response is always "missing" — and the still-running drug sims get
# killed mid-flight.
#
# Fix: for multi-drug runs the baseline pass saves to a *temporary* simLabel
# (`<simLabel>_tmp`) that the orchestrator ignores. Only after every condition
# has run does rank 0 assemble the complete dict (baseline pkl contents +
# drug_simData) and publish the real `<simLabel>_data.pkl` ONCE, via an atomic
# os.replace. The orchestrator therefore never sees a partial pkl and never
# scores/kills early. Baseline-only runs (no --drugs) keep the original
# single-write behaviour exactly.
# ---------------------------------------------------------------------------

_multidrug = bool(_drug_list)
_baseline_label = f"{_sim_label}_tmp" if _multidrug else _sim_label

_conditions = [("baseline", None)] + [(d, d) for d in _drug_list]
drug_simData = {}

# ---------------------------------------------------------------------------
# Per-condition data: for EVERY network (baseline + each drug) persist a
# standalone <simLabel>_<cond>_data.pkl. The <cond> infix means these names
# never match the orchestrator's exact `<simLabel>_data.pkl` completion probe
# (netpyne/batch/utils.py:533), so they cannot trigger early scoring. Rank-0
# only and best-effort — a save failure never fails the trial.
#
# Plotting is NOT done here: fitnessFunc_v2 already renders the rich raster +
# network-activity figures during scoring (plot_fitness_comparison ->
# <trial>_fitness_plot.png for baseline; plot_drug_response -> <trial>_drug_<drug>.png
# per drug), gated on plot_sim. Those are the canonical per-network plots.
# ---------------------------------------------------------------------------

def _copy_pkl(src_name, dst_name):
    """Rank-0: copy an already-written pkl to a differently-named sibling."""
    if rank != 0:
        return
    try:
        import shutil
        src = os.path.join(_save_folder, src_name)
        dst = os.path.join(_save_folder, dst_name)
        if os.path.exists(src):
            shutil.copyfile(src, dst)
            print(f"[init.py] condition pkl saved -> {dst}", flush=True)
        else:
            print(f"[init.py] condition pkl source missing: {src}", flush=True)
    except Exception as e:
        print(f"[init.py] _copy_pkl failed ({src_name}->{dst_name}): "
              f"{type(e).__name__}: {e}", flush=True)

def _save_condition_pkl(cond_name):
    """Rank-0: write a full <simLabel>_<cond>_data.pkl for the current gathered
    network via sim.saveData (same keys/structure as the trial pkl). sim.saveData
    derives its path from sim.cfg.simLabel, so switch it only for the write, then
    restore. Safe to call on rank 0 alone: data is already gathered, so saveData
    takes no collective path."""
    if rank != 0:
        return
    try:
        out_label = f"{_sim_label}_{cond_name}"
        _prev_cfg = sim.cfg.simLabel
        _prev_scfg = simConfig.simLabel
        try:
            sim.cfg.simLabel = out_label
            simConfig.simLabel = out_label
            sim.saveData()
        finally:
            sim.cfg.simLabel = _prev_cfg
            simConfig.simLabel = _prev_scfg
        print(f"[init.py] condition pkl saved -> {out_label}_data.pkl", flush=True)
    except Exception as e:
        print(f"[init.py] _save_condition_pkl failed ({cond_name}): "
              f"{type(e).__name__}: {e}", flush=True)

for _cond_idx, (_cond_name, _drug_arg) in enumerate(_conditions):
    # apply_drug mutates simConfig.scale_* (kept for traceability / the saved
    # cfg record); apply_drug_to_netparams scales the already-built connParams
    # weights, which is what sim.create actually reads (the cfg scalers were
    # consumed at netParams BUILD time, so mutating them alone does nothing).
    _snapshot = apply_drug(simConfig, _drug_arg) if _drug_arg else {}
    _np_snapshot = apply_drug_to_netparams(netParams, _drug_arg) if _drug_arg else {}
    try:
        if _cond_idx == 0:
            # Baseline: full createSimulateAnalyze so simData/V_soma/t land in
            # the pkl. For multi-drug runs redirect the auto-save to the temp
            # simLabel so the orchestrator does not pick it up before the drug
            # passes finish (restored in the finally below).
            if _multidrug:
                simConfig.simLabel = _baseline_label
            # createSimulateAnalyze, but with the cool-down-coarsened run swapped
            # in for sim.simulate()'s runSim (create -> run -> gather -> analyze).
            sim.create(simConfig=simConfig, netParams=netParams)
            _run_sim_with_cooldown()
            sim.gatherData()
            sim.analyze()
            # Persist the baseline network as its own pkl. analyze() just wrote
            # <_baseline_label>_data.pkl (the temp pkl in multidrug mode); copy it
            # to a canonical baseline name so it survives the temp-pkl cleanup below.
            _copy_pkl(f"{_baseline_label}_data.pkl", f"{_sim_label}_baseline_data.pkl")
        else:
            # Drug pass: rebuild + simulate, no analyze (saves wall-clock).
            # NetPyNE's sim.create reinitialises the network so the new scaler
            # values propagate into NetCon weights (see netParams.py:245-246, 265).
            sim.create(simConfig=simConfig, netParams=netParams)
            _run_sim_with_cooldown()
            sim.gatherData()
            if rank == 0:
                drug_simData[_cond_name] = {
                    "spkt": list(sim.allSimData.get("spkt", [])),
                    "spkid": list(sim.allSimData.get("spkid", [])),
                }
            # Persist this drug network as its own pkl, like baseline.
            _save_condition_pkl(_cond_name)
    except BaseException as e:
        msg = f"{type(e).__name__}: {e}"
        print(f"[init.py rank {rank}] {_cond_name} sim failed: {msg}", flush=True)
        if _cond_idx == 0:
            # Baseline failed: existing crash-safety path. Sentinel + clean exit.
            # _write_sentinel uses the real _sim_label, so the orchestrator sees
            # the final pkl path immediately and scores it as a failure.
            if rank == 0:
                traceback.print_exc()
            _write_sentinel(reason=msg)
            sys.exit(0)
        # Drug failed: isolate to this drug so the rest of the candidate still scores.
        if rank == 0:
            drug_simData[_cond_name] = {"_failure": True, "_failure_reason": msg}
    finally:
        if _cond_idx == 0 and _multidrug:
            # Restore the real label so any downstream consumer that reads
            # simConfig.simLabel sees the canonical trial label again.
            simConfig.simLabel = _sim_label
        if _np_snapshot:
            restore_netparams(netParams, _np_snapshot)
        if _snapshot:
            restore_cfg(simConfig, _snapshot)

# ---------------------------------------------------------------------------
# Publish the final pkl ONCE, atomically, with drug_simData already inside it.
# For multi-drug runs the baseline data currently lives in the temp pkl; rank 0
# reads it, attaches drug_simData, writes the real pkl via os.replace (atomic
# rename — the orchestrator only ever sees the complete file), then removes the
# temp pkl. Baseline-only runs already wrote the final pkl directly during
# createSimulateAnalyze, so there is nothing to do here.
# ---------------------------------------------------------------------------
if rank == 0 and _multidrug:
    _tmp_path = os.path.join(_save_folder, f"{_baseline_label}_data.pkl")
    _final_path = os.path.join(_save_folder, f"{_sim_label}_data.pkl")
    if os.path.exists(_tmp_path):
        try:
            with open(_tmp_path, 'rb') as _f:
                _data = pickle.load(_f)
            _data['drug_simData'] = drug_simData
            _writing_path = _final_path + ".writing"
            with open(_writing_path, 'wb') as _f:
                pickle.dump(_data, _f)
            os.replace(_writing_path, _final_path)   # atomic publish
            try:
                os.remove(_tmp_path)
            except OSError:
                pass
            print(f"[init.py] published final pkl with drug_simData "
                  f"({list(drug_simData.keys())}) -> {_final_path}", flush=True)
        except Exception as _e:
            print(f"[init.py] failed to assemble final pkl: "
                  f"{type(_e).__name__}: {_e}", flush=True)
            # Last resort: publish the baseline-only temp pkl so the orchestrator
            # still sees a result (scored as drug 'missing') instead of hanging
            # until maxiter_wait.
            try:
                if not os.path.exists(_final_path):
                    os.replace(_tmp_path, _final_path)
            except OSError:
                pass
    else:
        print(f"[init.py] temp baseline pkl not found at {_tmp_path}; "
              f"cannot publish final pkl", flush=True)

# # After simulation, create spike_times_by_unit and save it
# if rank == 0:  # Only do this on the master rank
#     try:
#         # Get the saved data path
#         if hasattr(sim.cfg, 'saveFolder') and hasattr(sim.cfg, 'simLabel'):
#             data_file = os.path.join(sim.cfg.saveFolder, sim.cfg.simLabel + '_data.pkl')
            
#             # Load the saved data
#             import pickle
#             with open(data_file, 'rb') as f:
#                 data = pickle.load(f)
            
#             # Extract spike data
#             if 'simData' in data and 'spkt' in data['simData'] and 'spkid' in data['simData']:
#                 spike_times = np.array(data['simData']['spkt']) / 1000  # Convert to seconds
#                 spike_ids = np.array(data['simData']['spkid'])
                
#                 # Create spike_times_by_unit dictionary
#                 spike_times_by_unit = {}
#                 for unit_id in np.unique(spike_ids):
#                     spike_times_by_unit[int(unit_id)] = spike_times[spike_ids == unit_id]
                
#                 # Add spike_times_by_unit and sampling_rate to simData
#                 data['simData']['spike_times_by_unit'] = spike_times_by_unit
#                 data['simData']['spike_times'] = spike_times
                
#                 # Add sampling_rate - for simulated data, this is computed from dt
#                 # sampling_rate = 1000.0 / dt (where dt is in ms)
#                 if hasattr(sim.cfg, 'dt'):
#                     sampling_rate = 1000.0 / sim.cfg.dt  # Convert from ms to Hz
#                 else:
#                     sampling_rate = 'Not available - dt not found in simConfig'
                
#                 data['simData']['sampling_rate'] = sampling_rate
                
#                 # Save back to file
#                 with open(data_file, 'wb') as f:
#                     pickle.dump(data, f)
                
#                 print(f"Successfully added spike_times_by_unit and sampling_rate to {data_file}")
#                 if isinstance(sampling_rate, (int, float)):
#                     print(f"Sampling rate: {sampling_rate} Hz")
#             else:
#                 print("Warning: No spike data found in simData")
#     except Exception as e:
#         print(f"Error adding spike_times_by_unit and sampling_rate: {e}")
#         import traceback
#         traceback.print_exc()