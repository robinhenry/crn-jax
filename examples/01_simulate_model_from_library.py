"""Minimal example — simulate the repressilator model from the library."""

from pathlib import Path

import jax
import numpy as np

from crn_jax.models import repressilator, sample_trajectories
from crn_jax.plotting import plot_species_trajectories

N_REPLICATES = 1000

dataset = sample_trajectories(
    repressilator,
    key=jax.random.PRNGKey(0),
    x0=np.zeros((N_REPLICATES, len(repressilator.SPECIES))),
)

fig, _ = plot_species_trajectories(dataset, title="Repressilator")
fig.savefig(Path(__file__).parent / "example_01_repressilator.png", dpi=120)
