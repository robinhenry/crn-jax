"""Repressilator — Elowitz-Leibler synthetic oscillator (BIOMD0000000012).

State: three species ``[A, B, C]`` forming a single repressive ring:
C ⊣ A ⊣ B ⊣ C. With sufficient cooperativity and a sharp enough Hill
curve, the system limit-cycles.

Reactions
---------
    R0:  ∅ → A     at rate  β_A0 + β_A1 · K_Cⁿ / (K_Cⁿ + Cⁿ)     ν = (+1,  0,  0)
    R1:  A → ∅     at rate  δ_A · A                                ν = (-1,  0,  0)
    R2:  ∅ → B     at rate  β_B0 + β_B1 · K_Aⁿ / (K_Aⁿ + Aⁿ)     ν = ( 0, +1,  0)
    R3:  B → ∅     at rate  δ_B · B                                ν = ( 0, -1,  0)
    R4:  ∅ → C     at rate  β_C0 + β_C1 · K_Bⁿ / (K_Bⁿ + Bⁿ)     ν = ( 0,  0, +1)
    R5:  C → ∅     at rate  δ_C · C                                ν = ( 0,  0, -1)

A single shared Hill exponent ``n`` is used for all three nodes (matches
the BioModels source). Defaults give δ ≈ 0.347 (lifetime ≈ 3 min) and
β₁ chosen so the deterministic system limit-cycles.
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
    beta_A0: float = 0.03
    beta_A1: float = 29.97
    beta_B0: float = 0.03
    beta_B1: float = 29.97
    beta_C0: float = 0.03
    beta_C1: float = 29.97
    K_A: float = 40.0
    K_B: float = 40.0
    K_C: float = 40.0
    n: float = 2.0
    delta_A: float = 0.347
    delta_B: float = 0.347
    delta_C: float = 0.347

    @classmethod
    def easy(cls) -> "Params":
        return cls(
            beta_A0=0.03,
            beta_A1=29.97,
            beta_B0=0.03,
            beta_B1=29.97,
            beta_C0=0.03,
            beta_C1=29.97,
            K_A=40.0,
            K_B=40.0,
            K_C=40.0,
            n=2.0,
            delta_A=0.347,
            delta_B=0.347,
            delta_C=0.347,
        )

    @classmethod
    def hard(cls) -> "Params":
        return cls(
            beta_A0=0.03,
            beta_A1=20.0,
            beta_B0=0.03,
            beta_B1=20.0,
            beta_C0=0.03,
            beta_C1=20.0,
            K_A=40.0,
            K_B=40.0,
            K_C=40.0,
            n=1.5,
            delta_A=0.347,
            delta_B=0.347,
            delta_C=0.347,
        )


def propensities_fn(params: Params) -> Callable[[State, Array], Array]:
    def f(state: State, _u: Array) -> Array:
        A = state.x[0]
        B = state.x[1]
        C = state.x[2]
        repress_a_by_c = 1.0 - hill_function(C, params.K_C, params.n)
        repress_b_by_a = 1.0 - hill_function(A, params.K_A, params.n)
        repress_c_by_b = 1.0 - hill_function(B, params.K_B, params.n)
        return jnp.array(
            [
                params.beta_A0 + params.beta_A1 * repress_a_by_c,
                params.delta_A * A,
                params.beta_B0 + params.beta_B1 * repress_b_by_a,
                params.delta_B * B,
                params.beta_C0 + params.beta_C1 * repress_c_by_b,
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
    n_steps: int = 2000,
    dt: float = 0.1,
    a0_dist: tuple = ("uniform", 0.0, 100.0),
    b0_dist: tuple = ("uniform", 0.0, 100.0),
    c0_dist: tuple = ("uniform", 0.0, 100.0),
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
