"""Positive autoregulation — one-node Hill self-activation.

State: one species ``X``.

Reactions
---------
    R0:  ∅ → X     at rate  β₀ + β₁ · Xⁿ / (Kⁿ + Xⁿ)        ν = (+1,)
    R1:  X → ∅     at rate  δ · X                            ν = (-1,)

For sub-cooperative Hill (``n ≤ 1``) the equilibrium is monostable and
graded; cooperativity ``n ≥ 2`` plus low leakage gives switch-like
behaviour — see :mod:`~crn_jax.models.bistable` for the bistable regime.
"""

import dataclasses
from typing import Callable

import jax.numpy as jnp
from jax import Array

from ..kinetics import hill_function
from ._common import (
    State,
    make_apply_reaction,
)

SPECIES: tuple[str, ...] = ("X",)
_STOICH: tuple[tuple[int, ...], ...] = (
    (+1,),
    (-1,),
)


@dataclasses.dataclass(frozen=True)
class Params:
    beta_0: float = 0.01
    beta_1: float = 5.0
    K: float = 1.0
    n: float = 1.0
    delta: float = 1.0

    @classmethod
    def easy(cls) -> "Params":
        return cls()

    @classmethod
    def hard(cls) -> "Params":
        return cls(beta_1=8.0, n=2.0)


def propensities_fn(params: Params) -> Callable[[State, Array], Array]:
    def f(state: State, _u: Array) -> Array:
        X = state.x[0]
        activation = hill_function(X, params.K, params.n)
        return jnp.array(
            [
                params.beta_0 + params.beta_1 * activation,
                params.delta * X,
            ]
        )

    return f


apply_reaction = make_apply_reaction(_STOICH)
