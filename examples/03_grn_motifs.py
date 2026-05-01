"""Example 03 — using the standard GRN motifs library.

``crn_jax.motifs`` ships pre-built canonical reaction networks. Each motif
exports the same surface (``State``, ``Params``, ``propensities_fn``,
``apply_reaction``, ``simulate_dataset``) so dropping a different system
into an analysis pipeline is a one-line change.

This example simulates two systems side by side — the simple inducible
motif (1 species, 1 input) and the C1-FFL with AND gate (3 species,
1 input) — and plots a single trajectory from each.
"""

from pathlib import Path

import jax
import matplotlib.pyplot as plt

from crn_jax.motifs import ffl_and, inducible


def main() -> None:
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)

    # Inducible: drive at u = 25 (saturating) for 24 h.
    ds_ind = inducible.simulate_dataset(
        k1,
        n_replicates=4,
        n_steps=1440,
        dt=1.0,
        u_dist=("uniform", 25.0, 25.0),  # constant u
    )

    # FFL: drive at u = 25 for 200 min, starting from zero so the
    # AND-gate turn-on dynamics is visible.
    ds_ffl = ffl_and.simulate_dataset(
        k2,
        n_replicates=4,
        n_steps=2000,
        dt=0.1,
        u_dist=("uniform", 25.0, 25.0),
        x0_dist=("zero",),
        y0_dist=("zero",),
        z0_dist=("zero",),
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    for i in range(ds_ind.Xs.shape[0]):
        ax.step(ds_ind.times, ds_ind.Xs[i], where="post", lw=1.0)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("X (molecules)")
    ax.set_title("Inducible — Hill-modulated birth-death (u = 25)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    t = ds_ffl.times
    for i in range(min(2, ds_ffl.Xs.shape[0])):
        ax.step(
            t, ds_ffl.Xs[i], where="post", lw=1.0, color="tab:blue", alpha=0.6, label="X (LacI)" if i == 0 else None
        )
        ax.step(
            t,
            ds_ffl.Ys[i],
            where="post",
            lw=1.0,
            color="tab:orange",
            alpha=0.6,
            label="Y (intermediate)" if i == 0 else None,
        )
        ax.step(
            t,
            ds_ffl.Zs[i],
            where="post",
            lw=1.0,
            color="tab:green",
            alpha=0.6,
            label="Z (AND-gated output)" if i == 0 else None,
        )
    ax.set_xlabel("time (min)")
    ax.set_ylabel("molecules")
    ax.set_title("C1-FFL with AND gate (u = 25, all species start at 0)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("crn_jax.motifs — two pre-built systems, one call each", fontsize=12)
    fig.tight_layout()

    out_path = Path(__file__).parent / "example_03.png"
    fig.savefig(out_path, dpi=120)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
