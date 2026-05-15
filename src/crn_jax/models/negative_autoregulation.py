"""Negative autoregulation: one-node Hill-repressed self-feedback.

State: one species ``X``.

Reactions
---------
    R0:  ∅ → X     at rate  β₀ + β₁ · Kⁿ / (Kⁿ + Xⁿ)   ν = (+1,)
    R1:  X → ∅     at rate  δ · X                      ν = (-1,)

The deterministic equilibrium is the root of
``β₀ + β₁ · Kⁿ / (Kⁿ + Xⁿ) = δ · X``.

Default parameters
------------------
Canonical textbook case from Alon, Fig 3.3, rescaled into min⁻¹ with a
30-min protein half-life (fast-growth E. coli convention) and scaled up
10× in copy-number space to put ⟨X⟩ in the realistic E. coli TF range.
The dimensionless ratio β₁ / (δ · K) = 5 is preserved from Alon's
analysis. The Hill coefficient is n=2 (TF dimer binding), more
biologically realistic than Alon's n=1 analytical convenience.
Qualitatively, the speed-up result holds for any n ≥ 1.
    β₀ = 0
    β₁ = 50 · ln(2)/30 ≈ 1.1552   (10× Alon's dimensionless β = 5)
    K  = 10                       (10× Alon's K = 1; in molecule counts)
    n  = 2                        (TF dimer cooperativity)
    δ  = ln(2)/30      ≈ 0.02310  (protein half-life 30 min)

Steady state ⟨X⟩ ≈ 15.2 (solve β₁/δ · K^n/(K^n + X^n) = X numerically;
≈ 17.9 for n=1, smaller for n=2 because the Hill saturates earlier).

Response time speeds up faster than n=1; the qualitative result that
"NAR is ~5–7× faster than simple decay" still holds.

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
    beta_1: float = 1.1552
    K: float = 10.0
    n: float = 2.0
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
        repression = repressive_hill(X, params.K, params.n)
        return jnp.array(
            [
                params.beta_0 + params.beta_1 * repression,  # R0
                params.delta * X,  # R1
            ]
        )

    return f
