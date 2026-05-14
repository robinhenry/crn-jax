"""Negative autoregulation — one-node Hill-repressed self-feedback.

State: one species ``X``.

Reactions
---------
    R0:  ∅ → X     at rate  β₀ + β₁ · Kⁿ / (Kⁿ + Xⁿ)   ν = (+1,)
    R1:  X → ∅     at rate  δ · X                      ν = (-1,)

The deterministic equilibrium is the root of
``β₀ + β₁ · Kⁿ / (Kⁿ + Xⁿ) = δ · X``.

Sources
-------
* https://mcb111.org/w11/w11-lecture.html
* https://www.cs.helsinki.fi/u/lmsalmel/cmsb09/lectures/CompMSysBio2009-Lecture5.pdf
* https://www.weizmann.ac.il/mcb/alon/sites/mcb.UriAlon/files/madar-arabinosepaper.pdf
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
        repression = repressive_hill(X, params.K, params.n)
        return jnp.array(
            [
                params.beta_0 + params.beta_1 * repression,  # R0
                params.delta * X,  # R1
            ]
        )

    return f
