"""Negative autoregulation — one-node Hill-repressed self-feedback.

State: one species ``X``.

Reactions
---------
    R0:  ∅ → X     at rate  β₀ + β₁ · Kⁿ / (Kⁿ + Xⁿ)   ν = (+1,)
    R1:  X → ∅     at rate  δ · X                      ν = (-1,)

The deterministic equilibrium is the root of
``β₀ + β₁ · Kⁿ / (Kⁿ + Xⁿ) = δ · X``.

Default parameters
------------------
Canonical textbook case from Alon, Fig 3.3, rescaled into min⁻¹.
Alon's dimensionless `δ = 1` is scaled by setting `δ = ln(2)/60 ≈ 0.01155 min⁻¹`
(the 60-min protein half-life used by Thattai & van Oudenaarden for E. coli);
β₁ is rescaled by the same factor.
K is a count and n is dimensionless, so both are unchanged.
    β₀ = 0
    β₁ = 5 · ln(2)/60 ≈ 0.0578   (Alon's β = 5 in dimensionless units)
    K  = 1                       (count)
    n  = 1                       (Alon's analytical case)
    δ  = ln(2)/60 ≈ 0.01155       (protein half-life 60 min)

Steady state ⟨X⟩ = (−1 + √21) / 2 ≈ 1.79 (depends only on β₁/δ = 5 and K).
Response time T₁/₂ ≈ ln(2)/(2δ) ≈ 30 min — half the simple-decay response.

Sources
-------
* Alon U (2007). An Introduction to Systems Biology, Ch. 3.4 (Fig 3.3) —
  dimensionless parameters above; time-unit conversion described above.
"""

import dataclasses
from typing import Self

import jax.numpy as jnp
from jax import Array

from ..kinetics import repressive_hill
from ..types import PropensitiesFn, SpeciesNames, State, StoichiometryMatrix
from ._common import make_apply_reaction


@dataclasses.dataclass(frozen=True)
class Params:
    beta_0: float = 0.0
    beta_1: float = 0.05776
    K: float = 1.0
    n: float = 1.0
    delta: float = 0.01155

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
        repression = repressive_hill(X, params.K, params.n)
        return jnp.array(
            [
                params.beta_0 + params.beta_1 * repression,  # R0
                params.delta * X,  # R1
            ]
        )

    return f
