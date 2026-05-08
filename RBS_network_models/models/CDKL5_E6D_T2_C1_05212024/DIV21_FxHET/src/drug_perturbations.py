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
