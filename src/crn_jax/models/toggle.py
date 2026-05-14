"""Toggle switch — Gardner-Cantor-Collins mutual-inhibition circuit (2000).

State: two species ``[A, B]``; each represses the other. We use
A = LacI and B = TetR to map onto the Lugagne 2017 *E. coli* synthetic
toggle.

Reactions
---------
    R0:  ∅ → A     at rate  β_A0 + β_A1 · K_B^n_B / (K_B^n_B + B^n_B)   ν = (+1,  0)
    R1:  A → ∅     at rate  δ_A · A                                     ν = (-1,  0)
    R2:  ∅ → B     at rate  β_B0 + β_B1 · K_A^n_A / (K_A^n_A + A^n_A)   ν = ( 0, +1)
    R3:  B → ∅     at rate  δ_B · B                                     ν = ( 0, -1)

Default parameters
------------------
Simplified version of the **Lugagne et al. (2017) Nature Communications**
8-reaction model, originally fitted to single-cell measurements of an *E. coli*
LacI/TetR toggle. Their mRNA dynamics (5-min half-life) are eliminated
in favour of an effective protein production rate. The result is the simple
4-reaction Hill form, with parameters in **arbitrary units (a.u.)** matching the
fluorescence calibration in the original paper.

    β_A0 = κp_L · κm0_L / g^m_L  =  0.9726 · 0.032 / 0.1386  ≈  0.2246  (basal LacI rate)
    β_A1 = κp_L · κm_L  / g^m_L  =  0.9726 · 8.30  / 0.1386  ≈  58.23   (max LacI rate)
    β_B0 = κp_T · κm0_T / g^m_T  =  1.170  · 0.119 / 0.1386  ≈  1.005   (basal TetR rate)
    β_B1 = κp_T · κm_T  / g^m_T  =  1.170  · 2.06  / 0.1386  ≈  17.39   (max TetR rate)
    K_A  = θ_LacI  ≈  31.94                                             (LacI repressing-TetR threshold)
    K_B  = θ_TetR  =  30.00                                             (TetR repressing-LacI threshold)
    n_A  = η_LacI  =  2                                                 (LacI Hill — dimer)
    n_B  = η_TetR  =  2                                                 (TetR Hill — dimer)
    δ_A  = g^p_L   =  0.0165 / min                                      (LacI half-life ≈ 42 min)
    δ_B  = g^p_T   =  0.0165 / min                                      (TetR half-life ≈ 42 min)

Three deterministic fixed points (asymmetric):
    "LacI wins" (stable):  A ≈ 660,  B ≈ 63    (a.u.)
    "TetR wins" (stable):  A ≈ 18,   B ≈ 864   (a.u.)
    Saddle:                A ≈ 73,   B ≈ 228   (a.u.)

The basins are asymmetric because the leak rates β_A0 ≈ 0.22 and
β_B0 ≈ 1.0 set non-trivial floor levels of the losing species, which
then partially repress the winner. The LacI promoter is stronger than
the TetR promoter, but the larger TetR leak compensates.


Sources
-------
* Lugagne JB, Sosa Carrillo S, Kirch M, Köhler A, Batt G, Hersen P
  (2017). Balancing a genetic toggle switch by real-time feedback
  control and periodic forcing. Nature Communications 8:1671 —
  canonical modern E. coli LacI/TetR toggle with fitted parameters.
* Gardner TS, Cantor CR, Collins JJ (2000). Construction of a genetic
  toggle switch in Escherichia coli. Nature 403:339–342 — the original
  synthetic-biology landmark.
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
    beta_A0: float = 0.2246
    beta_A1: float = 58.23
    beta_B0: float = 1.005
    beta_B1: float = 17.39
    K_A: float = 31.94
    K_B: float = 30.00
    n_A: float = 2.0
    n_B: float = 2.0
    delta_A: float = 0.0165
    delta_B: float = 0.0165

    @classmethod
    def default(cls) -> Self:
        return cls()


SPECIES: SpeciesNames = ("A", "B")
_STOICHIOMETRY: StoichiometryMatrix = (
    (+1, 0),  # R0: ∅ → A
    (-1, 0),  # R1: A → ∅
    (0, +1),  # R2: ∅ → B
    (0, -1),  # R3: B → ∅
)
apply_reaction = make_apply_reaction(_STOICHIOMETRY)


def propensities_fn(params: Params) -> PropensitiesFn:
    def f(state: State, _u: Array) -> Array:
        A = state.x[0]
        B = state.x[1]
        repress_a_by_b = repressive_hill(B, params.K_B, params.n_B)
        repress_b_by_a = repressive_hill(A, params.K_A, params.n_A)
        return jnp.array(
            [
                params.beta_A0 + params.beta_A1 * repress_a_by_b,  # R0
                params.delta_A * A,  # R1
                params.beta_B0 + params.beta_B1 * repress_b_by_a,  # R2
                params.delta_B * B,  # R3
            ]
        )

    return f
