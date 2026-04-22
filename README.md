# crn-jax

Chemical reaction networks in JAX — a tiny, GPU-parallel Gillespie / Stochastic Simulation Algorithm (SSA) library.

## Install

```bash
pip install crn-jax
# or, from source:
pip install git+https://github.com/robinhenry/crn-jax
# for local development:
git clone https://github.com/robinhenry/crn-jax && cd crn-jax && pip install -e ".[test,examples]"
```

Depends on `jax` / `jaxlib` only. `matplotlib` is pulled in via the `examples` extra for the plotting helper.

## Quickstart

A 1-species birth-death process, `∅ → X` at rate λ and `X → ∅` at rate μ·x, simulated for 10 independent replicates and plotted:

```python
from typing import NamedTuple
import jax, jax.numpy as jnp
from crn_jax import simulate_trajectory, plot_trajectories

BIRTH_RATE, DEATH_RATE = 3.0, 0.1    # steady-state mean λ/μ = 30

class State(NamedTuple):
    time: jax.Array
    x: jax.Array
    next_reaction_time: jax.Array    # carried across intervals

def propensities(s, _action):
    return jnp.array([BIRTH_RATE, DEATH_RATE * s.x])

def apply_reaction(s, j):
    return s._replace(x=s.x + jnp.where(j == 0, 1.0, -1.0))

state0 = State(jnp.array(0.0), jnp.array(0.0), jnp.array(jnp.inf))

@jax.jit
@jax.vmap
def run_one(key):
    return simulate_trajectory(
        key=key,
        initial_state=state0,
        timestep=1.0,
        n_steps=200,
        compute_propensities_fn=propensities,
        apply_reaction_fn=apply_reaction,
    )

states = run_one(jax.random.split(jax.random.PRNGKey(0), 10))
times = jnp.arange(1, 201) * 1.0

fig, ax = plot_trajectories(times, states.x, ylabel="X (molecules)")
ax.axhline(BIRTH_RATE / DEATH_RATE, color="k", ls="--", label="λ/μ")
fig.savefig("birth_death.png")
```

See the [examples](examples/) folder for the full version plus an optogenetic Hill-regulated gene-expression demo with per-replicate action schedules.

## Key features

- **Exact SSA** — pure-JAX implementation of the Gillespie algorithm for chemical reaction networks.
- **JIT-compiled** — the entire loop compiles under `jax.jit`.
- **GPU-parallel** — 1 000+ independent trajectories on a single GPU under `jax.vmap`, with no Python overhead per step.
- **Discretisation-safe** — pending reaction times are preserved across simulation-interval boundaries, so trajectories are physically correct under RL-style stepping.
- **Bring-your-own state** — the loop operates on any PyTree (NamedTuple, Flax struct dataclass, Equinox module, …).

## API

```python
# Main entry point: scan n_steps fixed-length intervals, stack the per-step states.
from crn_jax import simulate_trajectory

# Finer control: one interval at a time (RL-style), or until an absolute time.
from crn_jax.gillespie import simulate_interval, simulate_until

# Plotting helper: step-plots a single trajectory or an (N, T) ensemble.
from crn_jax import plot_trajectories

# Optional kinetic-law helpers.
from crn_jax.kinetics import hill_function, sample_lognormal
```

| function              | when to reach for it                                                          |
| --------------------- | ----------------------------------------------------------------------------- |
| `simulate_trajectory` | You want a full trajectory on a fixed sampling grid. Start here.              |
| `simulate_interval`   | You're driving the system yourself, one step at a time (e.g. an RL rollout).  |
| `simulate_until`      | You need a custom state shape or a non-uniform time grid. Fully generic.      |
| `plot_trajectories`   | Quick look at the output.                                                     |

`simulate_trajectory` and `simulate_interval` assume the state exposes `time`, `next_reaction_time`, and a `_replace` method (`NamedTuple`, Flax struct dataclass, Equinox module, …). `simulate_until` has no such requirement — you pass `get_time_fn` / `update_time_fn` callbacks instead.

## License

MIT. See [LICENSE](LICENSE).