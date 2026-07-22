"""Profile multi-drug simulation cost: rank scaling + build-vs-simulate split.

Run under NEURON/MPI, e.g.::

    srun -n 32 nrniv -mpi -python benchmark_drug_sim.py

All configuration comes from environment variables (so our flags never collide
with the args sim.readCmdLineArgs parses):

    FEATURE_DATA_PATH   experimental target h5 (sets network size ~533 cells).
    T_TARGET_S          sim window seconds (default 10 -> duration 15 s w/ cooldown).
    BENCH_MODE          'A' = create()+simulate() per condition (current init.py),
                        'B' = create() once, in-place NetCon rescale + re-simulate.
    BENCH_DRUGS         comma-separated drug names (default 'bicuculline,ap5_nbqx').
    BENCH_CFG_JSON      a real trial_*_cfg.json whose params seed a bursting net,
                        sourced into __main__.params so cfg.py picks them up.

Emits CSV lines on rank 0 prefixed 'BENCH,' that run_benchmark_sweep.sh collects:

    BENCH,<mode>,<ranks>,<cond>,<create_s>,<simulate_s>,<n_spikes>

mode A and mode B produce the same network/weights per condition, so for each
condition the n_spikes MUST match between modes -- that is the correctness gate
proving the in-place rescale is a faithful drop-in for the per-drug sim.create().
"""
import os
import sys
import json
import time

# --- network model location -------------------------------------------------
_SRC = ("/pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/"
        "RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/src")
_CFG_PY = os.path.join(_SRC, "cfg.py")
_NETPARAMS_PY = os.path.join(_SRC, "netParams.py")
# make `import drug_perturbations` resolve the same module init.py uses
sys.path.insert(0, _SRC)

# --- benchmark config from env ----------------------------------------------
_T_TARGET = os.environ.setdefault("T_TARGET_S", "10")
_MODE = os.environ.get("BENCH_MODE", "A").strip().upper()
_DRUGS = [d.strip() for d in os.environ.get(
    "BENCH_DRUGS", "bicuculline,ap5_nbqx").split(",") if d.strip()]
_CFG_JSON = os.environ.get("BENCH_CFG_JSON", "").strip()
if not os.environ.get("FEATURE_DATA_PATH"):
    raise SystemExit("FEATURE_DATA_PATH must point at the experimental target h5")

# --- seed `params` from a real candidate so the network bursts realistically.
# cfg.py does `from __main__ import params` first (cfg.py:23) and sets each as a
# cfg attribute; we expose only the physiological knobs netParams/cfg consume.
_PARAM_KEYS = [
    # connectivity / synaptic
    'probEE', 'probEI', 'probIE', 'probII', 'probLengthConst', 'propVelocity',
    'weightEE_AMPA', 'weightEE_NMDA', 'weightEI_AMPA', 'weightEI_NMDA',
    'weightIE_GABA', 'weightII_GABA',
    'tau1_AMPA', 'tau2_AMPA', 'tau1_NMDA', 'tau2_NMDA', 'tau1_GABA', 'tau2_GABA',
    # active conductances
    'gnabar_E', 'gnabar_I', 'gkbar_E', 'gkbar_I',
    'gnabar_E_std', 'gnabar_I_std', 'gkbar_E_std', 'gkbar_I_std',
    # passive cell properties (read by netParams.init_normally_distributed_properties)
    'E_L_mean', 'E_L_stdev', 'E_Ra_mean', 'E_Ra_stdev', 'E_diam_mean', 'E_diam_stdev',
    'I_L_mean', 'I_L_stdev', 'I_Ra_mean', 'I_Ra_stdev', 'I_diam_mean', 'I_diam_stdev',
]
params = {}
if _CFG_JSON:
    with open(_CFG_JSON) as _f:
        _d = json.load(_f)
    _sc = _d.get('simConfig', _d)
    params = {k: _sc[k] for k in _PARAM_KEYS if k in _sc}

# ---------------------------------------------------------------------------
from netpyne import sim
from neuron import h

try:
    from mpi4py import MPI
    _RANKS = MPI.COMM_WORLD.Get_size()
    _RANK = MPI.COMM_WORLD.Get_rank()
except ImportError:
    _RANKS, _RANK = 1, 0

from drug_perturbations import (DRUG_REGISTRY, _SYNMECH_TO_SCALER,
                                apply_drug_to_netparams, restore_netparams)

# scale_GABA -> 'GABA', etc. (invert the synMech->scaler map)
_SCALER_TO_SYNMECH = {v: k for k, v in _SYNMECH_TO_SCALER.items()}


def _syn_scalers_for(drug):
    """Drug name -> {synMech_label: scale_factor} for in-place NetCon scaling."""
    out = {}
    for scaler_attr, val in DRUG_REGISTRY[drug]['cfg_overrides'].items():
        mech = _SCALER_TO_SYNMECH.get(scaler_attr)
        if mech is None:
            raise NotImplementedError(
                f"{drug}: scaler {scaler_attr} is not a synaptic-mech scaler "
                f"(in-place rescale only handles {sorted(_SCALER_TO_SYNMECH)})")
        out[mech] = float(val)
    return out


def _rescale_netcons(scalers):
    """Multiply each matching NetCon weight in place; return restore snapshot.

    Walks this rank's local cells (sim.net.cells is rank-local under MPI), so the
    scaling is applied everywhere without any collective.
    """
    snap = []
    for cell in sim.net.cells:
        for conn in cell.conns:
            mech = conn.get('synMech')
            if mech in scalers:
                nc = conn.get('hObj')
                if nc is None:
                    continue
                old = nc.weight[0]
                snap.append((nc, old))
                nc.weight[0] = old * scalers[mech]
    return snap


def _restore_netcons(snap):
    for nc, old in snap:
        nc.weight[0] = old


def _n_spikes():
    """Total spike count across all ranks (valid on rank 0 after gatherData)."""
    return len(sim.allSimData.get('spkt', [])) if _RANK == 0 else 0


def _emit(cond, create_s, sim_s, nspk):
    if _RANK == 0:
        print(f"BENCH,{_MODE},{_RANKS},{cond},{create_s:.3f},{sim_s:.3f},{nspk}",
              flush=True)


# --- load the model, mirroring sim.readCmdLineArgs (setup.py:270-303) but
# injecting our candidate params onto cfg BETWEEN loading cfg.py and netParams.py.
# This matters because: (1) cfg.py only setattr's params in its except branch
# (default evol_params, which lacks the cell-property knobs like E_L_mean), and
# (2) netParams.py reads `from __main__ import cfg` at load time and immediately
# consumes cfg.E_L_mean etc. In production NetPyNE Batch supplies these via
# cmdline overrides; here we set them directly so the network matches the chosen
# candidate.
import __main__
cfgModule = sim.loadPythonModule(_CFG_PY)
simConfig = cfgModule.cfg
for _k, _v in params.items():
    setattr(simConfig, _k, _v)
__main__.cfg = simConfig
netParamsModule = sim.loadPythonModule(_NETPARAMS_PY)
netParams = netParamsModule.netParams

# strip disk I/O so timing reflects compute, not pickling (~70 MB/sim otherwise)
simConfig.savePickle = False
simConfig.saveJson = False
simConfig.saveDataInclude = []
simConfig.saveFolder = os.environ.get("BENCH_TMP", "/tmp")
simConfig.simLabel = "bench"

_conditions = ["baseline"] + _DRUGS

if _RANK == 0:
    print(f"[bench] mode={_MODE} ranks={_RANKS} T_target={_T_TARGET}s "
          f"duration={simConfig.duration}ms cells={simConfig.num_excite}E+"
          f"{simConfig.num_inhib}I conditions={_conditions}", flush=True)

_wall0 = time.perf_counter()

if _MODE == "A":
    # Current init.py behaviour: full rebuild per condition.
    for cond in _conditions:
        np_snap = apply_drug_to_netparams(netParams, cond) if cond != "baseline" else {}
        t0 = time.perf_counter()
        sim.create(simConfig=simConfig, netParams=netParams)
        t1 = time.perf_counter()
        sim.simulate()          # = runSim + gatherData
        t2 = time.perf_counter()
        _emit(cond, t1 - t0, t2 - t1, _n_spikes())
        if np_snap:
            restore_netparams(netParams, np_snap)

elif _MODE == "B":
    # Build once, then in-place NetCon rescale + re-simulate per drug.
    t0 = time.perf_counter()
    sim.create(simConfig=simConfig, netParams=netParams)
    t1 = time.perf_counter()
    sim.simulate()
    t2 = time.perf_counter()
    _emit("baseline", t1 - t0, t2 - t1, _n_spikes())

    for drug in _DRUGS:
        scalers = _syn_scalers_for(drug)
        snap = _rescale_netcons(scalers)
        ta = time.perf_counter()
        sim.setupRecording()    # reset spike/trace recorders for a clean re-run
        sim.runSim()
        sim.gatherData()
        tb = time.perf_counter()
        _emit(drug, 0.0, tb - ta, _n_spikes())
        _restore_netcons(snap)

else:
    raise SystemExit(f"unknown BENCH_MODE {_MODE!r} (expected 'A' or 'B')")

if _RANK == 0:
    print(f"BENCH_TOTAL,{_MODE},{_RANKS},{time.perf_counter() - _wall0:.3f}",
          flush=True)

h.quit()
