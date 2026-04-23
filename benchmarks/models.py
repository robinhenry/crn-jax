"""Reference reaction-network models used across benchmark backends.

Each backend (crn-jax, GillesPy2, StochPy, …) implements the same model
natively in whatever API it exposes; the shared spec here just pins down the
parameters, initial state, and sampling grid so results are comparable.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BirthDeath:
    """1-species birth-death process.

    Reactions
        ∅  -- λ   -->  X      (birth)
        X  -- μ·x -->  ∅      (death)

    Steady state is Poisson(λ/μ).
    """

    birth_rate: float = 3.0      # λ
    death_rate: float = 0.1      # μ
    initial_x: float = 0.0
    dt: float = 1.0              # sampling interval
    n_steps: int = 200           # number of sampling intervals

    @property
    def t_final(self) -> float:
        return self.dt * self.n_steps

    @property
    def steady_state_mean(self) -> float:
        return self.birth_rate / self.death_rate


BIRTH_DEATH = BirthDeath()
