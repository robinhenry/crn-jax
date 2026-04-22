# crn-jax

Chemical reaction networks in JAX — a tiny, GPU-parallel Stochastic
Simulation Algorithm (SSA) library.

> **Status:** `0.1.0` — Gillespie SSA only. τ-leaping, next-reaction method,
> and a Chemical Langevin Equation driver are on the roadmap below.

```python
import jax, jax.numpy as jnp
from typing import NamedTuple
from crn_jax import run_gillespie_loop

class State(NamedTuple):
    time: jax.Array
    x: jax.Array

def propensities(s, action):                 # [birth, death]
    return jnp.array([1.0, 0.1 * s.x])

def apply_reaction(s, j):
    return s._replace(x=s.x + jnp.where(j == 0, 1.0, -1.0))

state = State(time=jnp.array(0.0), x=jnp.array(0.0))
final, _ = run_gillespie_loop(
    key=jax.random.PRNGKey(0),
    initial_state=state,
    action=jnp.array(0.0),
    target_time=100.0,
    max_steps=10_000,
    compute_propensities_fn=propensities,
    apply_reaction_fn=apply_reaction,
    get_time_fn=lambda s: s.time,
    update_time_fn=lambda s, t: s._replace(time=t),
)
print(int(final.x))                          # ~10 at steady state
```

That's the entire public API for running a simulation. Everything else is
optional sugar.

## What this package is (and isn't)

`crn-jax` is a **pure-JAX implementation of exact stochastic simulation** for
chemical reaction networks. The core loop:

- compiles under `jax.jit`,
- parallelises under `jax.vmap` (1 000+ independent trajectories on a single
  GPU, with no Python overhead per step),
- preserves pending reaction times across simulation-interval boundaries, so
  trajectories are physically correct under RL-style discretisation.

It is deliberately **not**:

- a full systems-biology framework (no ODE integrators, no stoichiometry DSL,
  no visualisation — those are library concerns, not algorithm concerns),
- a replacement for `gillespy2`, `StochSS`, `COPASI`, etc. if you want a
  batteries-included workflow on CPU.

The target audience is researchers who already use JAX and want an SSA
primitive they can drop into their own models.

## Install

```bash
pip install crn-jax                 # once published
# or, from source:
pip install git+https://github.com/robinhenry/crn-jax
# for local development:
git clone https://github.com/robinhenry/crn-jax && cd crn-jax && pip install -e ".[test,examples]"
```

Depends on `jax` / `jaxlib` only.

## API surface

```python
from crn_jax import run_gillespie_loop          # core SSA driver
from crn_jax.extras import hill_function, sample_lognormal
from crn_jax.templates import step_interval      # fixed-dt RL-style wrapper
```

### `run_gillespie_loop(key, initial_state, action, target_time, max_steps, compute_propensities_fn, apply_reaction_fn, get_time_fn, update_time_fn, pending_reaction_time=None, previous_action=None)`

Advances a state forward to `target_time` under the exact Gillespie SSA.
Returns `(final_state, next_reaction_time)`.

You supply:

| callable | signature | what it does |
|---|---|---|
| `compute_propensities_fn` | `(state, action) -> Array[M]` | non-negative reaction rates |
| `apply_reaction_fn` | `(state, reaction_idx) -> state` | applies reaction `j` |
| `get_time_fn` | `state -> Array` | reads scalar time |
| `update_time_fn` | `(state, time) -> state` | writes scalar time |

The state object is whatever you want — a NamedTuple, a Flax struct dataclass,
an Equinox module, a plain PyTree. The loop never inspects its contents.

See the [examples](examples/) folder for worked 1-species, opto-Hill, and
parallel-parameter-sweep demos.

## Examples

```bash
pip install -e ".[examples]"
python examples/01_birth_death.py
python examples/02_opto_hill_1d.py
python examples/03_vmap_parameter_sweep.py
python examples/04_step_interval.py
```

## Roadmap

The package name is `crn-jax` (chemical reaction networks) rather than
`gillespie-jax` because the natural next additions to the library are:

- `crn_jax.tau_leap.run_tau_leap_loop` — approximate SSA for stiff regimes.
- `crn_jax.next_reaction.run_anderson_loop` — Anderson's modified
  next-reaction method.
- `crn_jax.cle.run_cle_loop` — Chemical Langevin Equation via
  Euler-Maruyama (continuous diffusion approximation).
- `crn_jax.network.Stoichiometry` — convenience object for building large
  reaction networks from a stoichiometry matrix + rate-law callables.

These will land behind feature flags rather than as breaking API changes.

## Citing

If you use this in academic work, please cite the original Gillespie paper:

> Gillespie, D. T. (1977). *Exact stochastic simulation of coupled chemical
> reactions.* **Journal of Physical Chemistry**, 81(25), 2340-2361.

## License

MIT. See [LICENSE](LICENSE).
