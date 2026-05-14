"""Positive autoregulation — one-node Hill self-activation.

State: one species ``X``.

Reactions
---------
    R0:  ∅ → X     at rate  β₀ + β₁ · Xⁿ / (Kⁿ + Xⁿ)   ν = (+1,)
    R1:  X → ∅     at rate  δ · X                      ν = (-1,)

For sub-cooperative Hill (``n ≤ 1``) the equilibrium is monostable and
graded; cooperativity ``n ≥ 2`` plus appropriate β₀, β₁/δK ratios give a
**bistable** switch-like regime with two stable fixed points (low / high)
separated by a saddle. Both regimes are the same reaction network — only
the parameters differ. Use :meth:`Params.default` for the graded case and
:meth:`Params.bistable` for the bistable regime.

Default parameters (monostable, n=1)
--------------------------------
Adapted from Alon, Fig 3.5 (n=1 analytical case), rescaled into min⁻¹ via
δ = ln(2)/60 (matching ``single_gene``'s 60-min protein half-life). Same
time-unit calibration as ``negative_autoregulation``, but the leak β₀ is
deliberately set higher than the textbook β₀ = 0 to keep the stochastic
model from getting trapped at the absorbing state X = 0 (see note below).
    β₀ = 0.5 · ln(2)/60 ≈ 5.776e-3    (mean wait at X=0 ≈ 173 min)
    β₁ = 5   · ln(2)/60 ≈ 0.0578
    K  = 1                            (count)
    n  = 1                            (Alon's analytical case)
    δ  = ln(2)/60        ≈ 0.01155    (protein half-life 60 min)

Deterministic steady state ⟨X⟩ ≈ 4.6 (vs Alon's β₀=0 case of 4.0).
Response time T₁/₂ ~ 2/δ ≈ 173 min — slower than the simple-decay 60 min.

Bistable parameters (Params.bistable)
-------------------------------------
Taken from the Caltech BE150 / Bi 250b notes (Bois & Elowitz),
with K and β₁ scaled up 5× from the textbook values to push the high
state from X ≈ 10 (where it is *metastable* under stochastic dynamics —
fluctuations cross the saddle and commit to the absorbing X=0) up to
X ≈ 50 (where the barrier in molecule-count space is wide enough to be
stochastically persistent). Dimensionless ratios β₁/(δK) = 2.5 and
n = 4 are preserved exactly.
    β₀ = 0                          (no basal leak — X=0 is absorbing)
    β₁ = 50 · ln(2)/60 ≈ 0.5776
    K  = 20                         (count, 5× textbook 4)
    n  = 4                          (cooperativity; ≥2 needed for bistability)
    δ  = ln(2)/60       ≈ 0.01155
Deterministic fixed points: X = 0 (stable), X_saddle ≈ 17, X_high ≈ 49.

Because β₀ = 0, the low fixed point in the stochastic model is exactly
X = 0 (zero propensity at X=0 — no escape). This is faithful to the
textbook deterministic treatment: replicates starting below the saddle
fall to and remain at zero; replicates above commit to the high basin
and remain there over the simulation horizon.

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
    beta_0: float = 0.005776
    beta_1: float = 0.05776
    K: float = 1.0
    n: float = 1.0
    delta: float = 0.01155

    @classmethod
    def default(cls) -> Self:
        return cls()

    @classmethod
    def bistable(cls) -> Self:
        """Bistable regime — BE150 ratios (Bois & Elowitz, Caltech), with K
        and β₁ scaled up 5× to give a stochastically-persistent high state
        at X ≈ 50 (rather than X ≈ 10, where the saddle is too close in
        count space and the high state escapes to the absorbing X=0).
        Three deterministic fixed points: X=0 (stable, absorbing),
        X_saddle ≈ 17, X_high ≈ 49.
        """
        return cls(
            beta_0=0.0,
            beta_1=0.5776,  # = 50 · ln(2)/60
            K=20.0,
            n=4.0,
            delta=0.01155,
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
