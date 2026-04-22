# crn-jax examples

Each file is a self-contained, runnable script. They share no code beyond what
they import from `crn_jax`.

| script | what it shows |
|---|---|
| [`01_birth_death.py`](01_birth_death.py) | Minimal 1-species birth-death process. The Gillespie SSA in ~60 LoC. Plots the time-course and the empirical distribution at steady state. |
| [`02_opto_hill_1d.py`](02_opto_hill_1d.py) | Gene expression driven by a Hill-function light input. Shows how to parameterise propensities with a control input (action). |
| [`03_vmap_parameter_sweep.py`](03_vmap_parameter_sweep.py) | Run 1 024 independent trajectories in parallel with different kinetic parameters via `jax.vmap` + `jax.jit`. The "why this exists" demo. |
| [`04_step_interval.py`](04_step_interval.py) | Using `crn_jax.templates.step_interval` to advance a state one fixed timestep at a time (RL-style loops). |

Install with the example dependencies:

```bash
pip install -e ".[examples]"
python examples/01_birth_death.py
```

All examples print summary statistics to stdout. Those that plot (01, 02)
save figures to the current directory as `example_XX.png`.
