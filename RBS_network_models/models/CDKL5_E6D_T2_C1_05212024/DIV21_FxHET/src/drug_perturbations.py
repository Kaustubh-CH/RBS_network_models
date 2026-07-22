"""Hand-coded drug perturbation registry for the multi-drug optimization flow.

Each registry entry maps a drug name (CLI flag value, e.g. ``--drugs bicuculline``)
to the cfg attribute overrides that simulate that drug. The cfg already exposes
pharmacological scalers at cfg.py:116-119 (scale_AMPA, scale_NMDA, scale_GABA,
scale_K) which are applied in netParams.py:245-246, 265 -- so applying a drug
is just a matter of mutating those scalers before calling sim.create / sim.simulate.

The registry is the single source of truth for two consumers:

  * src/init.py            - mutates the live simConfig per condition before
                             running each candidate simulation.
  * _scripts/extract_drug_effects.py (optionally) - records the param overrides
                             into the experimental target h5 attrs so the link
                             between an experimental drug condition and its
                             simulated counterpart is traceable.

Per the cfg.py:110-115 comment block, the canonical drug -> scaler mappings are::

    AP5 + NBQX               -> scale_AMPA = 0, scale_NMDA = 0
    AP5                      -> scale_NMDA = 0
    Bicuculline / Gabazine   -> scale_GABA = 0
    4AP                      -> scale_K    = 0

To add a new drug: append a new entry below. To change an existing one:
edit ``cfg_overrides`` -- both the simulator and the experimental ratio
extractor (if invoked with --drug_perturbations_module) will pick it up.
"""

from typing import Dict, Tuple


DRUG_REGISTRY: Dict[str, Dict] = {
    "bicuculline": {
        "cfg_overrides": {"scale_GABA": 0.0},
        "description": "Competitive GABA-A antagonist; full block at saturating dose.",
    },
    "gabazine": {
        "cfg_overrides": {"scale_GABA": 0.0},
        "description": "Selective GABA-A antagonist; full block.",
    },
    "ap5_nbqx": {
        "cfg_overrides": {"scale_AMPA": 0.0, "scale_NMDA": 0.0},
        "description": "AP5 (NMDA block) + NBQX (AMPA block). Glutamatergic transmission silenced.",
    },
    "ap5": {
        "cfg_overrides": {"scale_NMDA": 0.0},
        "description": "Selective NMDA receptor antagonist.",
    },
    "fourap": {
        "cfg_overrides": {"scale_K": 0.0},
        "description": "4-aminopyridine; broad Kv channel block. Increases excitability.",
    },
}


def validate_drug(drug_name: str) -> None:
    """Raise ValueError with the list of registered names if drug_name isn't known."""
    if drug_name not in DRUG_REGISTRY:
        raise ValueError(
            f"Unknown drug {drug_name!r}. Registered drugs: "
            f"{sorted(DRUG_REGISTRY.keys())}"
        )


def get_overrides(drug_name: str) -> Dict[str, float]:
    """Return a fresh copy of the cfg_overrides dict for the named drug."""
    validate_drug(drug_name)
    return dict(DRUG_REGISTRY[drug_name]["cfg_overrides"])


def apply_drug(cfg, drug_name: str) -> Dict[str, Tuple[float, float]]:
    """Mutate ``cfg`` in place with the drug's overrides.

    Returns a dict mapping each overridden attribute to ``(prev_value, new_value)``
    so the caller can restore the cfg after the per-condition simulation finishes
    (init.py uses this to keep baseline and drug runs isolated within a single
    Python process).
    """
    overrides = get_overrides(drug_name)
    snapshot: Dict[str, Tuple[float, float]] = {}
    for attr, new_val in overrides.items():
        if not hasattr(cfg, attr):
            raise AttributeError(
                f"cfg has no attribute {attr!r}; cannot apply drug {drug_name!r}. "
                "Check that the cfg defines pharmacological scalers (cfg.py:116-119)."
            )
        prev = getattr(cfg, attr)
        setattr(cfg, attr, new_val)
        snapshot[attr] = (prev, new_val)
    return snapshot


def restore_cfg(cfg, snapshot: Dict[str, Tuple[float, float]]) -> None:
    """Undo a previous apply_drug -- restore each attribute to its pre-drug value."""
    for attr, (prev, _new) in snapshot.items():
        setattr(cfg, attr, prev)


# ---------------------------------------------------------------------------
# netParams.connParams weight scaling (the mechanism that actually bites)
#
# netParams.py bakes cfg.scale_AMPA / scale_NMDA / scale_GABA into CONCRETE
# connParams weights at build time (netParams.py:245-246, 265), e.g.
#     E->*:  'weight': [w_AMPA * cfg.scale_AMPA, w_NMDA * cfg.scale_NMDA]  # list
#     I->*:  'weight':  w_GABA * cfg.scale_GABA                            # scalar
# Because netParams is built ONCE (cfg.scale_*=1.0) and reused, mutating
# cfg.scale_* before sim.create has NO effect -- the baked floats are what
# sim.create reads. To make a drug bite we therefore scale those baked
# connParams weights directly, matched by synMech, before sim.create, and
# restore them afterwards so the next condition starts from baseline.
# Connectivity itself is unchanged (deterministic NetPyNE seeds + HDF5
# positions) so baseline and drug networks differ only in synaptic weight.
# ---------------------------------------------------------------------------

# synMech label -> cfg scaler attribute that scales that mechanism's weight.
_SYNMECH_TO_SCALER: Dict[str, str] = {
    'AMPA': 'scale_AMPA',
    'NMDA': 'scale_NMDA',
    'GABA': 'scale_GABA',
}


def apply_drug_to_netparams(netParams, drug_name: str) -> Dict[str, object]:
    """Scale the already-built connParams weights for a drug, in place.

    For each connParam whose synMech is targeted by one of the drug's scalers,
    multiply the corresponding weight slot by the scaler value. Returns a
    snapshot ``{connKey: original_weight}`` (deep-copied) for
    ``restore_netparams``.

    Raises NotImplementedError for non-synaptic scalers (e.g. ``scale_K`` /
    4AP, which acts on cellParams gkbar, not connParams) so an unsupported
    drug fails loudly instead of silently doing nothing.
    """
    import copy

    overrides = get_overrides(drug_name)
    syn_scalers = {s: v for s, v in overrides.items()
                   if s in _SYNMECH_TO_SCALER.values()}
    other_scalers = {s: v for s, v in overrides.items() if s not in syn_scalers}
    if other_scalers:
        raise NotImplementedError(
            f"Drug {drug_name!r} uses non-synaptic scaler(s) {sorted(other_scalers)}; "
            f"apply_drug_to_netparams only handles synaptic conductance scalers "
            f"({sorted(_SYNMECH_TO_SCALER.values())}). scale_K (4AP) scales "
            f"cellParams gkbar (netParams.py:168) and needs separate handling."
        )

    snapshot: Dict[str, object] = {}
    for key, cp in netParams.connParams.items():
        mechs = cp.get('synMech')
        w = cp.get('weight')
        if mechs is None or w is None:
            continue
        if isinstance(mechs, (list, tuple)):
            if not any(_SYNMECH_TO_SCALER.get(m) in syn_scalers for m in mechs):
                continue
            snapshot[key] = copy.deepcopy(w)
            new_w = list(w)
            for i, m in enumerate(mechs):
                scaler = _SYNMECH_TO_SCALER.get(m)
                if scaler in syn_scalers and i < len(new_w):
                    new_w[i] = w[i] * syn_scalers[scaler]
            cp['weight'] = new_w
        else:
            scaler = _SYNMECH_TO_SCALER.get(mechs)
            if scaler in syn_scalers:
                snapshot[key] = copy.deepcopy(w)
                cp['weight'] = w * syn_scalers[scaler]
    return snapshot


def restore_netparams(netParams, snapshot: Dict[str, object]) -> None:
    """Undo apply_drug_to_netparams -- restore each connParam weight."""
    for key, orig_w in snapshot.items():
        netParams.connParams[key]['weight'] = orig_w
