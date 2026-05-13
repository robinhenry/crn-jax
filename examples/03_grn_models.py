"""Example 03 — using the standard GRN models library.

``crn_jax.models`` ships a library of canonical reaction networks (see
``src/crn_jax/models/library.json``). Each model exports the same surface
(``Params`` with ``.easy()`` / ``.hard()`` factories, ``propensities_fn``,
``apply_reaction``, ``simulate_dataset``) so dropping a different system
into an analysis pipeline is a one-line change.

This example plots two visually distinct models side by side: the
Elowitz-Leibler repressilator (sustained oscillation under negative
ring feedback) and the Gardner-Cantor-Collins toggle switch (mutual
inhibition giving rise to two stable basins).
"""

from pathlib import Path

import jax
import matplotlib.pyplot as plt

from crn_jax.models import repressilator, toggle


def main() -> None:
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)

    # Repressilator: easy regime → sustained oscillation (n=2, β₁=29.97).
    ds_rep = repressilator.simulate_dataset(
        k1,
        params=repressilator.Params.easy(),
        n_replicates=3,
        n_steps=2000,
        dt=0.1,
    )

    # Toggle switch: BIOMD0000000507 params, broad IC sampling so some
    # replicates land in the A-high basin and others in the B-high basin.
    ds_tog = toggle.simulate_dataset(
        k2,
        params=toggle.Params.easy(),
        n_replicates=8,
        n_steps=2000,
        dt=0.05,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    colours = ["tab:blue", "tab:orange", "tab:green"]

    # --- Repressilator ---
    ax = axes[0]
    for i in range(ds_rep.xs.shape[0]):
        for j, name in enumerate(ds_rep.species):
            ax.step(
                ds_rep.times,
                ds_rep.xs[i, :, j],
                where="post",
                lw=1.0,
                color=colours[j],
                alpha=0.4,
                label=name if i == 0 else None,
            )
    ax.set_xlabel("time")
    ax.set_ylabel("molecules")
    ax.set_title("Repressilator — A ⊣ B ⊣ C ⊣ A (easy regime, n=2)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # --- Toggle switch ---
    ax = axes[1]
    for i in range(ds_tog.xs.shape[0]):
        for j, name in enumerate(ds_tog.species):
            ax.step(
                ds_tog.times,
                ds_tog.xs[i, :, j],
                where="post",
                lw=1.0,
                color=colours[j],
                alpha=0.4,
                label=name if i == 0 else None,
            )
    ax.set_xlabel("time")
    ax.set_ylabel("molecules")
    ax.set_title("Toggle switch — mutual inhibition (BIOMD0000000507 params)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("crn_jax.models — two pre-built systems, one call each", fontsize=12)
    fig.tight_layout()

    out_path = Path(__file__).parent / "example_03.png"
    fig.savefig(out_path, dpi=120)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
