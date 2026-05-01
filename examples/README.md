# crn-jax examples

Each file is a self-contained, runnable script.

| script | what it shows |
|---|---|
| [`01_birth_death.py`](01_birth_death.py) | A minimal 1-species birth-death process. Simulates 10 trajectories with `run_trajectory` + `vmap` (over keys) and plots them against the steady-state mean λ/μ. |
| [`02_opto_hill_1d.py`](02_opto_hill_1d.py) | An optogenetic-driven gene expression process with fully-independent replicates: each trajectory has its own key, initial x₀, and input light step-on time. This is essentially illustrating running a batch of independent experiments via `jax.vmap` over keys, initial states, and per-replicate action schedules in a single call. |
| [`03_grn_motifs.py`](03_grn_motifs.py) | Using the `crn_jax.motifs` library: pull two pre-built systems (inducible Hill expression and a 3-species C1-FFL with AND gate) and simulate each with a single `simulate_dataset(...)` call. Demonstrates the canonical-motif convenience API. |

Install with the example dependencies:

```bash
pip install -e ".[examples]"
python examples/01_birth_death.py
```

Both examples save figures to the current directory as `example_XX.png`.
