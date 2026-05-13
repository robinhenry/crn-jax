"""Toggle switch — Gardner-Cantor-Collins mutual-inhibition motif (2000).

State: two species ``[A, B]``; each represses the other.

Reactions
---------
    R0:  ∅ → A     at rate  β_A0 + β_A1 · K_B^n_B / (K_B^n_B + B^n_B)   ν = (+1,  0)
    R1:  A → ∅     at rate  δ_A · A                                       ν = (-1,  0)
    R2:  ∅ → B     at rate  β_B0 + β_B1 · K_A^n_A / (K_A^n_A + A^n_A)   ν = ( 0, +1)
    R3:  B → ∅     at rate  δ_B · B                                       ν = ( 0, -1)

The easy regime uses the published BIOMD0000000507 parameters
(``n_B = 2.5`` — non-integer Hill on purpose, matching the BioModels
source); the hard regime moves closer to the switching boundary.
"""

import dataclasses
import functools
from typing import Callable

import jax.numpy as jnp
from jax import Array

from ..kinetics import hill_function
from ..types import PRNGKey
from ._common import (
    Dataset,
    State,
    make_apply_reaction,
    make_vmap_simulator,
    run_dataset,
)

SPECIES: tuple[str, ...] = ("A", "B")
_STOICH: tuple[tuple[int, ...], ...] = (
    (+1, 0),
    (-1, 0),
    (0, +1),
    (0, -1),
)


@dataclasses.dataclass(frozen=True)
class Params:
    beta_A0: float = 15.6
    beta_A1: float = 156.25
    beta_B0: float = 0.0
    beta_B1: float = 15.6
    K_A: float = 2.0015
    K_B: float = 2.9618e-05
    n_A: float = 1.0
    n_B: float = 2.5
    delta_A: float = 1.0
    delta_B: float = 1.0

    @classmethod
    def easy(cls) -> "Params":
        return cls()

    @classmethod
    def hard(cls) -> "Params":
        return cls(
            beta_A0=10.0,
            beta_A1=100.0,
            beta_B1=10.0,
            K_A=1.5,
            K_B=0.01,
            n_A=2.0,
            n_B=2.0,
        )


def propensities_fn(params: Params) -> Callable[[State, Array], Array]:
    def f(state: State, _u: Array) -> Array:
        A = state.x[0]
        B = state.x[1]
        repress_a_by_b = 1.0 - hill_function(B, params.K_B, params.n_B)
        repress_b_by_a = 1.0 - hill_function(A, params.K_A, params.n_A)
        return jnp.array(
            [
                params.beta_A0 + params.beta_A1 * repress_a_by_b,
                params.delta_A * A,
                params.beta_B0 + params.beta_B1 * repress_b_by_a,
                params.delta_B * B,
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
    n_steps: int = 2000,
    dt: float = 0.05,
    a0_dist: tuple = ("uniform", 0.0, 200.0),
    b0_dist: tuple = ("uniform", 0.0, 200.0),
) -> Dataset:
    return run_dataset(
        key,
        species=SPECIES,
        simulator=_build_simulator(n_steps, params),
        n_replicates=n_replicates,
        n_steps=n_steps,
        dt=dt,
        x0_dists=(a0_dist, b0_dist),
    )
