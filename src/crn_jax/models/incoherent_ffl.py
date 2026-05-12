"""Incoherent feed-forward loop — A activates B; A activates C; B represses C.

State: three species ``[A, B, C]``. The opposing direct and indirect paths
from A to C generate a pulse / adaptive response on C.

Reactions
---------
    R0:  ∅ → A     at rate  α_A                                       ν = (+1,  0,  0)
    R1:  A → ∅     at rate  δ_A · A                                   ν = (-1,  0,  0)
    R2:  ∅ → B     at rate  β_B0 + β_B1 · A^n_A / (K_A^n_A + A^n_A)   ν = ( 0, +1,  0)
    R3:  B → ∅     at rate  δ_B · B                                   ν = ( 0, -1,  0)
    R4:  ∅ → C     at rate  β_C0 + β_C1 · A^n_A / (K_A^n_A + A^n_A)
                                  · K_B^n_B / (K_B^n_B + B^n_B)        ν = ( 0,  0, +1)
    R5:  C → ∅     at rate  δ_C · C                                   ν = ( 0,  0, -1)
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

SPECIES: tuple[str, ...] = ("A", "B", "C")
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
    alpha_A: float = 5.0
    delta_A: float = 1.0
    beta_B0: float = 0.05
    beta_B1: float = 5.0
    K_A: float = 1.0
    n_A: float = 1.0
    delta_B: float = 1.0
    beta_C0: float = 0.05
    beta_C1: float = 5.0
    K_B: float = 1.0
    n_B: float = 1.0
    delta_C: float = 1.0

    @classmethod
    def easy(cls) -> "Params":
        return cls(
            alpha_A=5.0,
            delta_A=1.0,
            beta_B0=0.05,
            beta_B1=5.0,
            K_A=1.0,
            n_A=1.0,
            delta_B=1.0,
            beta_C0=0.05,
            beta_C1=5.0,
            K_B=1.0,
            n_B=1.0,
            delta_C=1.0,
        )

    @classmethod
    def hard(cls) -> "Params":
        return cls(
            alpha_A=1.0,
            delta_A=1.0,
            beta_B0=0.01,
            beta_B1=5.0,
            K_A=1.0,
            n_A=2.0,
            delta_B=1.0,
            beta_C0=0.01,
            beta_C1=5.0,
            K_B=1.0,
            n_B=2.0,
            delta_C=1.0,
        )


def propensities_fn(params: Params) -> Callable[[State, Array], Array]:
    def f(state: State, _u: Array) -> Array:
        A = state.x[0]
        B = state.x[1]
        C = state.x[2]
        activate_b_by_a = hill_function(A, params.K_A, params.n_A)
        repress_c_by_b = 1.0 - hill_function(B, params.K_B, params.n_B)
        return jnp.array(
            [
                params.alpha_A,
                params.delta_A * A,
                params.beta_B0 + params.beta_B1 * activate_b_by_a,
                params.delta_B * B,
                params.beta_C0 + params.beta_C1 * activate_b_by_a * repress_c_by_b,
                params.delta_C * C,
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
    n_steps: int = 1000,
    dt: float = 0.05,
    a0_dist: tuple = ("uniform", 0.0, 10.0),
    b0_dist: tuple = ("uniform", 0.0, 10.0),
    c0_dist: tuple = ("uniform", 0.0, 10.0),
) -> Dataset:
    return run_dataset(
        key,
        species=SPECIES,
        simulator=_build_simulator(n_steps, params),
        n_replicates=n_replicates,
        n_steps=n_steps,
        dt=dt,
        x0_dists=(a0_dist, b0_dist, c0_dist),
    )
