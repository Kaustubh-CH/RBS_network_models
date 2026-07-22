"""Parameter sampling in unit space and conversion to physical space.

Unit-space convention
---------------------
Every tuneable parameter is represented in a **unit space** spanning [-1, +1].
This provides a uniform, dimensionless coordinate system that is independent of
the physical units or scale of individual parameters (e.g. conductances in S/cm²
vs. time constants in ms).

Linear rescaling formulae
~~~~~~~~~~~~~~~~~~~~~~~~~
*Unit → Physical*::

    physical = ((unit + 1) / 2) * (upper - lower) + lower

*Physical → Unit*::

    unit = 2 * (physical - lower) / (upper - lower) - 1

where ``lower`` and ``upper`` are the physical bounds for a given parameter.

Sampling strategies
~~~~~~~~~~~~~~~~~~~
* **Uniform random** (``use_lhs=False``, default) – fast and simple; each
  coordinate is drawn independently from U(-1, +1).  Suitable for large sample
  counts where coverage is less critical.
* **Latin Hypercube Sampling** (``use_lhs=True``) – provides better space-filling
  properties by ensuring each parameter's marginal distribution is stratified.
  Recommended when the number of samples is small relative to the dimensionality
  of the parameter space.
"""

from typing import Dict, List, Optional

import numpy as np
from scipy.stats.qmc import LatinHypercube


def sample_unit_params(
    n_samples: int,
    n_params: int,
    seed: Optional[int] = None,
    use_lhs: bool = False,
) -> np.ndarray:
    """Sample parameter vectors in unit space [-1, +1].

    Parameters
    ----------
    n_samples : int
        Number of parameter vectors to generate.
    n_params : int
        Dimensionality of the parameter space.
    seed : int, optional
        Random seed for reproducibility.
    use_lhs : bool, optional
        If *True*, use Latin Hypercube Sampling for better space-filling
        coverage.  If *False* (default), draw i.i.d. uniform samples.

    Returns
    -------
    np.ndarray
        Array of shape ``[n_samples, n_params]`` with values in [-1, +1],
        dtype float32.
    """
    if use_lhs:
        sampler = LatinHypercube(d=n_params, seed=seed)
        # LHS produces samples in [0, 1]; rescale to [-1, +1]
        samples = sampler.random(n=n_samples)
        unit_params = 2.0 * samples - 1.0
    else:
        rng = np.random.default_rng(seed)
        unit_params = rng.uniform(-1.0, 1.0, size=(n_samples, n_params))

    return unit_params.astype(np.float32)


def unit_to_physical(unit_params: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Convert parameters from unit space [-1, +1] to physical space.

    Parameters
    ----------
    unit_params : np.ndarray
        Array of shape ``[N, P]`` with values in [-1, +1].
    bounds : np.ndarray
        Array of shape ``[P, 2]`` where each row is ``[lower, upper]``.

    Returns
    -------
    np.ndarray
        Array of shape ``[N, P]`` in physical space, dtype float32.
    """
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    physical = ((unit_params + 1.0) / 2.0) * (upper - lower) + lower
    # Clip to bounds to handle floating-point edge cases
    physical = np.clip(physical, lower, upper)
    return physical.astype(np.float32)


def physical_to_unit(physical_params: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Convert parameters from physical space to unit space [-1, +1].

    This is the inverse of :func:`unit_to_physical`.

    Parameters
    ----------
    physical_params : np.ndarray
        Array of shape ``[N, P]`` in physical space.
    bounds : np.ndarray
        Array of shape ``[P, 2]`` where each row is ``[lower, upper]``.

    Returns
    -------
    np.ndarray
        Array of shape ``[N, P]`` with values in [-1, +1], dtype float32.
    """
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    unit = 2.0 * (physical_params - lower) / (upper - lower) - 1.0
    unit = np.clip(unit, -1.0, 1.0)
    return unit.astype(np.float32)


def physical_to_cfg_dict(
    physical_row: np.ndarray, param_names: List[str]
) -> Dict[str, float]:
    """Convert a single physical-parameter vector to a config dictionary.

    The returned dictionary is ready to be injected into a NetPyNE ``cfg``
    object via ``setattr``.

    Parameters
    ----------
    physical_row : np.ndarray
        1-D array of length ``P`` containing physical parameter values.
    param_names : list of str
        Ordered list of parameter names matching the columns of
        ``physical_row``.

    Returns
    -------
    dict
        Flat dictionary ``{param_name: float_value}`` with native Python
        floats (not numpy scalars).
    """
    return {name: float(val) for name, val in zip(param_names, physical_row)}
