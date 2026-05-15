from typing import Callable, NamedTuple

import numpy as np
from jax import Array

PRNGKey = Array

SpeciesNames = tuple[str, ...]
StoichiometryMatrix = tuple[tuple[int, ...], ...]


class State(NamedTuple):
    """State carried through the Gillespie SSA loop.

    This class works for every model: ``x`` is always a 1-D
    ``(n_species,)`` JAX array, even for single-species models, which keeps
    :func:`sample_trajectories` uniform across the library. ``x[i]`` is the
    count of the i-th species named in the model's ``SPECIES`` tuple.
    """

    time: Array
    x: Array
    next_reaction_time: Array


# Closure returned by a model's ``propensities_fn(params)``: maps
# ``(state, u)`` to a length-``n_reactions`` array of per-reaction rates.
# ``u`` is the input.
PropensitiesFn = Callable[[State, Array], Array]


# --- Output shape ------------------------------------------------------------


class Dataset(NamedTuple):
    """Output of every model's :func:`sample_trajectories`.

    All arrays are NumPy (host-side). The per-species trajectory tensor
    ``xs`` and the flat one-step transition arrays ``X_t`` and ``dX`` make
    both views available without recomputation.
    """

    times: np.ndarray  # (n_steps,) — sample times in `dt` units
    species: tuple[str, ...]  # species names, len == n_species
    x0: np.ndarray  # (n_replicates, n_species) — initial counts supplied by caller
    xs: np.ndarray  # (n_replicates, n_steps, n_species) — full trajectories
    X_t: np.ndarray  # (n_replicates * n_steps, n_species) — flat per-step state
    dX: np.ndarray  # (n_replicates * n_steps, n_species) — flat one-step delta
