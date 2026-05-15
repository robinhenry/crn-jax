"""A birth-death stochastic process.

State: one species ``X``.

Reactions
---------
    R0:  ∅ → X     at rate  α                ν = (+1,)
    R1:  X → ∅     at rate  δ · X            ν = (-1,)

Stationary distribution is Poisson with mean ``⟨X⟩ = α / δ``.

Default parameters
------------------
Rates in min⁻¹. ``⟨X⟩ = 50`` puts the stationary distribution in the
typical E. coli transcription-factor copy-number range (tens to
hundreds), well above the Poisson noise floor (σ = √50 ≈ 7).
Decay rate set to a 30-min half-life (typical doubling time for fast-growth
E. coli).

    α = 50 · ln(2)/30 ≈ 1.1552   (per-min birth rate)
    δ = ln(2)/30      ≈ 0.02310  (per-mol decay, 30-min half-life)
"""

import dataclasses
from typing import Self

import jax.numpy as jnp
from jax import Array

from ..types import PropensitiesFn, SpeciesNames, State, StoichiometryMatrix
from ._common import make_apply_reaction


@dataclasses.dataclass(frozen=True)
class Params:
    alpha: float = 1.1552
    delta: float = 0.02310

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
        return jnp.array(
            [
                params.alpha,  # R0
                params.delta * X,  # R1
            ]
        )

    return f
