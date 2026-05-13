"""Telegraph promoter — two-state bursting model for single-cell expression.

State: three species ``[S, M, P]`` where ``S ∈ {0, 1}`` is the promoter
state (off / on), ``M`` is mRNA count, and ``P`` is protein count.

Reactions
---------
    R0:  S=0 → S=1         at rate  k_on · (1 − S)         ν = (+1,  0,  0)
    R1:  S=1 → S=0         at rate  k_off · S              ν = (-1,  0,  0)
    R2:  S=1 → S=1 + M     at rate  β · S                  ν = ( 0, +1,  0)
    R3:  M       → ∅       at rate  γ_M · M                ν = ( 0, -1,  0)
    R4:  M       → M + P   at rate  k_tl · M               ν = ( 0,  0, +1)
    R5:  P       → ∅       at rate  δ_P · P                ν = ( 0,  0, -1)

The promoter switching propensities are zero outside ``S ∈ {0, 1}`` by
construction, so ``S`` is invariant in that set provided the initial
condition is binary (use the default ``("bernoulli", 0.5)`` IC).
"""

import dataclasses
import functools
from typing import Callable

import jax.numpy as jnp
from jax import Array

from ..types import PRNGKey
from ._common import (
    Dataset,
    State,
    make_apply_reaction,
    make_vmap_simulator,
    run_dataset,
)

SPECIES: tuple[str, ...] = ("S", "M", "P")
_STOICH: tuple[tuple[int, ...], ...] = (
    (+1, 0, 0),
    (-1, 0, 0),
    (0, +1, 0),
    (0, -1, 0),
    (0, 0, +1),
    (0, 0, -1),
)


@dataclasses.dataclass(frozen=True)
class Params:
    k_on: float = 1.0
    k_off: float = 1.0
    beta: float = 5.0
    gamma_M: float = 1.0
    k_tl: float = 10.0
    delta_P: float = 0.2

    @classmethod
    def easy(cls) -> "Params":
        return cls()

    @classmethod
    def hard(cls) -> "Params":
        return cls(k_on=0.2, k_off=0.2, beta=10.0)


def propensities_fn(params: Params) -> Callable[[State, Array], Array]:
    def f(state: State, _u: Array) -> Array:
        S = state.x[0]
        M = state.x[1]
        P = state.x[2]
        return jnp.array(
            [
                params.k_on * (1.0 - S),
                params.k_off * S,
                params.beta * S,
                params.gamma_M * M,
                params.k_tl * M,
                params.delta_P * P,
            ]
        )

    return f


apply_reaction = make_apply_reaction(_STOICH)


@functools.lru_cache(maxsize=None)
def _build_simulator(n_steps: int, params: Params):
    return make_vmap_simulator(n_steps, propensities_fn(params), apply_reaction)


def simulate_dataset(
    key: PRNGKey,
    *,
    params: Params = Params(),
    n_replicates: int = 256,
    n_steps: int = 1500,
    dt: float = 0.1,
    s0_dist: tuple = ("bernoulli", 0.5),
    m0_dist: tuple = ("uniform", 0.0, 5.0),
    p0_dist: tuple = ("uniform", 0.0, 150.0),
) -> Dataset:
    return run_dataset(
        key,
        species=SPECIES,
        simulator=_build_simulator(n_steps, params),
        n_replicates=n_replicates,
        n_steps=n_steps,
        dt=dt,
        x0_dists=(s0_dist, m0_dist, p0_dist),
    )
