"""Positive autoregulation: one-node Hill self-activation.

State: one species ``X``.

Reactions
---------
    R0:  ∅ → X     at rate  β₀ + β₁ · Xⁿ / (Kⁿ + Xⁿ)   ν = (+1,)
    R1:  X → ∅     at rate  δ · X                      ν = (-1,)

For sub-cooperative Hill (``n ≤ 1``) the equilibrium is monostable and
graded; cooperativity ``n ≥ 2`` plus appropriate β₀, β₁/δK ratios give a
*bistable* switch-like regime with two stable fixed points (low / high)
separated by a saddle. Both regimes are the same reaction network — only
the parameters differ.

Use :meth:`Params.default` for the graded case and :meth:`Params.bistable`
for the bistable regime.

Default parameters (monostable)
-------------------------------
Adapted from Alon, Fig 3.5, rescaled into min⁻¹ with a 30-min protein
half-life (fast-growth E. coli convention) and scaled up 10× in
copy-number space so that ⟨X⟩ ≈ 51 sits in the typical E. coli TF range.
The Hill coefficient is n=2 (TF dimer cooperativity), more biologically
realistic than Alon's n=1 analytical convenience. The leak β₀ is set to
5% of β₁ (typical for E. coli promoters, deliberately above Alon's
β₀ = 0 so the stochastic model doesn't get trapped at X = 0). Alon's
dimensionless ratio β₁/(δ·K) = 5 is preserved.
    β₀ = 2.5 · ln(2)/30  ≈ 0.0578     (5% leak; mean wait at X=0 ≈ 17 min)
    β₁ = 50  · ln(2)/30  ≈ 1.1552
    K  = 10                           (10× Alon's K = 1; molecule counts)
    n  = 2                            (TF dimer cooperativity)
    δ  = ln(2)/30        ≈ 0.02310    (protein half-life 30 min)

Deterministic steady state ⟨X⟩ ≈ 50.7 (slight upward shift from
Alon-n=1's 46 because n=2 saturates the Hill more sharply at ⟨X⟩ > K).
Response time still slower than simple-decay 30 min half-life.

Bistable parameters (Params.bistable)
-------------------------------------
Taken from the Caltech BE150 / Bi 250b notes (Bois & Elowitz), with K
and β₁ scaled up 5× to push the high state up to X ≈ 51 (stochastically
persistent), 30-min protein half-life (fast-growth E. coli), and a 5%
basal leak (β₀/β₁) so cells in the low basin can recover from X=0 via
spontaneous expression. Dimensionless ratio β₁/(δK) = 2.5 and n = 4 are
preserved exactly, β₀/β₁ = 0.05 is a typical E. coli promoter leak.
    β₀ = 2.5 · ln(2)/30 ≈ 0.0578     (5% leak; mean wait at X=0 ≈ 17 min)
    β₁ = 50  · ln(2)/30 ≈ 1.1552
    K  = 20                          (count, 5× textbook 4)
    n  = 4                           (cooperativity; ≥2 needed for bistability)
    δ  = ln(2)/30        ≈ 0.02310

Deterministic fixed points: X_low ≈ 2.5 (stable), X_saddle ≈ 15,
X_high ≈ 51 (stable).

The low basin (X ≈ 2.5) replaces the strict-X=0 absorbing state from
BE150's β₀=0 case — cells in the low basin fluctuate around ~2.5 rather
than being permanently stuck. This is more biologically faithful (real
promoters always leak slightly) and avoids the absorbing-state
issue under stochastic dynamics. Cells that fluctuate to X=0
recover in ~17 min (well under the 30-min half-life).

Sources
-------
* Alon U (2007). An Introduction to Systems Biology, Ch. 3.5 (Fig 3.5) —
  monostable-regime parameters, time-unit conversion as described above.
* Bois JS, Elowitz MB. Biological Circuit Design (BE 150 / Bi 250b),
  Caltech — bistability example uses β=10, γ=1, K=4, n=4 in dimensionless
  units. https://biocircuits.github.io/
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
    beta_0: float = 0.0578
    beta_1: float = 1.1552
    K: float = 10.0
    n: float = 2.0
    delta: float = 0.02310

    @classmethod
    def default(cls) -> Self:
        return cls()

    @classmethod
    def bistable(cls) -> Self:
        return cls(
            beta_0=0.0578,  # = 2.5 · ln(2)/30; 5% leak, mean wait at X=0 ≈ 17 min
            beta_1=1.1552,  # = 50 · ln(2)/30
            K=20.0,
            n=4.0,
            delta=0.02310,
        )


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
