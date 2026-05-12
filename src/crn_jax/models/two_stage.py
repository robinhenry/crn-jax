"""Two-stage transcription-translation — mRNA + protein with hidden-state lag.

State: two species ``[M, P]`` (mRNA, protein).

Reactions
---------
    R0:  ∅       → M       at rate  α_M             ν = (+1,  0)
    R1:  M       → ∅       at rate  γ_M · M          ν = (-1,  0)
    R2:  M       → M + P   at rate  k_tl · M         ν = ( 0, +1)
    R3:  P       → ∅       at rate  δ_P · P          ν = ( 0, -1)
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

SPECIES: tuple[str, ...] = ("M", "P")
_STOICH: tuple[tuple[int, ...], ...] = (
    (+1, 0),
    (-1, 0),
    (0, +1),
    (0, -1),
)


@dataclasses.dataclass(frozen=True)
class Params:
    alpha_M: float = 5.0
    gamma_M: float = 1.0
    k_tl: float = 10.0
    delta_P: float = 0.2

    @classmethod
    def easy(cls) -> "Params":
        return cls(alpha_M=5.0, gamma_M=1.0, k_tl=10.0, delta_P=0.2)

    @classmethod
    def hard(cls) -> "Params":
        return cls(alpha_M=1.0, gamma_M=1.0, k_tl=5.0, delta_P=0.5)


def propensities_fn(params: Params) -> Callable[[State, Array], Array]:
    def f(state: State, _u: Array) -> Array:
        M = state.x[0]
        P = state.x[1]
        return jnp.array(
            [
                params.alpha_M,
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
    m0_dist: tuple = ("uniform", 0.0, 10.0),
    p0_dist: tuple = ("uniform", 0.0, 300.0),
) -> Dataset:
    return run_dataset(
        key,
        species=SPECIES,
        simulator=_build_simulator(n_steps, params),
        n_replicates=n_replicates,
        n_steps=n_steps,
        dt=dt,
        x0_dists=(m0_dist, p0_dist),
    )
