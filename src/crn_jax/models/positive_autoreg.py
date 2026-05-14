"""Positive autoregulation — one-node Hill self-activation.

State: one species ``X``.

Reactions
---------
    R0:  ∅ → X     at rate  β₀ + β₁ · Xⁿ / (Kⁿ + Xⁿ)   ν = (+1,)
    R1:  X → ∅     at rate  δ · X                      ν = (-1,)

For sub-cooperative Hill (``n ≤ 1``) the equilibrium is monostable and
graded; cooperativity ``n ≥ 2`` plus low leakage gives switch-like
behaviour — see :mod:`~crn_jax.models.bistable` for the bistable regime.

Sources
-------
* https://mcb111.org/w11/w11-lecture.html
* https://www.cs.helsinki.fi/u/lmsalmel/cmsb09/lectures/CompMSysBio2009-Lecture5.pdf
"""

import dataclasses
from typing import Self

import jax.numpy as jnp
from jax import Array

from ..kinetics import hill_function
from ..types import PropensitiesFn, SpeciesNames, State, StoichiometryMatrix
from ._common import make_apply_reaction


@dataclasses.dataclass(frozen=True)
class Params:
    beta_0: float = 0.01
    beta_1: float = 5.0
    K: float = 1.0
    n: float = 1.0
    delta: float = 1.0

    @classmethod
    def default(cls) -> Self:
        return cls()


SPECIES: SpeciesNames = ("X",)
_STOICHIOMETRY: StoichiometryMatrix = (
    (+1,),  # R0: ∅ → X
    (-1,),  # R1: X → ∅
)
apply_reaction = make_apply_reaction(_STOICHIOMETRY)


def propensities_fn(params: Params) -> PropensitiesFn:
    def f(state: State, _u: Array) -> Array:
        X = state.x[0]
        activation = hill_function(X, params.K, params.n)
        return jnp.array(
            [
                params.beta_0 + params.beta_1 * activation,  # R0
                params.delta * X,  # R1
            ]
        )

    return f
