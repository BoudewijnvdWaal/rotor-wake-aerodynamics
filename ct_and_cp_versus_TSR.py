"""
Generate Ct and torque versus TSR from the BEM model and save the figure.
"""

from pathlib import Path
import contextlib
import io

import matplotlib.pyplot as plt
import numpy as np

from BEM_TG2303 import (
    U0,
    Radius,
    blades,
    RootLocation_R,
    TipLocation_R,
    polar_alpha,
    polar_cl,
    polar_cd,
    initialise,
    executeBEM,
)


def compute_ct_torque_vs_tsr(tsr_values, n_annuli=100):
    """Run BEM for each TSR and return Ct and torque arrays."""
    ct_values = np.zeros(len(tsr_values))
    torque_values = np.zeros(len(tsr_values))

    for i, tsr in enumerate(tsr_values):
        omega = tsr * U0 / Radius
        r_R, chord_distribution, twist_distribution, _, _ = initialise(n_annuli)

        # Keep this script focused on the final Ct/Cp graph only.
        with contextlib.redirect_stdout(io.StringIO()):
            ct, _, _, _, torque, _ = executeBEM(
                U0,
                tsr,
                RootLocation_R,
                TipLocation_R,
                omega,
                Radius,
                blades,
                r_R,
                chord_distribution,
                twist_distribution,
                polar_alpha,
                polar_cl,
                polar_cd,
                plot_results=False,
                output_dir=None,
            )

        ct_values[i] = ct
        torque_values[i] = torque

    return ct_values, torque_values


def plot_ct_torque_vs_tsr(tsr_values, ct_values, torque_values, output_file):
    """Create and save a dual-axis plot for torque and Ct versus TSR."""
    fig, ax_torque = plt.subplots(figsize=(10, 6))
    ax_ct = ax_torque.twinx()

    line_torque, = ax_torque.plot(
        tsr_values,
        torque_values,
        "o-",
        color="tab:blue",
        linewidth=1.8,
        markersize=6,
        label="Torque (Nm)",
    )
    line_ct, = ax_ct.plot(
        tsr_values,
        ct_values,
        "x--",
        color="tab:red",
        linewidth=1.8,
        markersize=6,
        label="Thrust Coefficient (C_t)",
    )

    ax_torque.set_title("Performance vs TSR")
    ax_torque.set_xlabel("Tip-Speed Ratio (TSR)")
    ax_torque.set_ylabel("Torque (Nm)", color="tab:blue")
    ax_ct.set_ylabel("Thrust Coefficient (C_t)", color="tab:red")

    ax_torque.tick_params(axis="y", labelcolor="tab:blue")
    ax_ct.tick_params(axis="y", labelcolor="tab:red")
    ax_torque.grid(True)

    ax_torque.legend([line_torque, line_ct], [line_torque.get_label(), line_ct.get_label()], loc="best")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    tsr_values = np.arange(4.0, 12.5, 0.5)
    ct_values, torque_values = compute_ct_torque_vs_tsr(tsr_values, n_annuli=100)

    output_file = Path("figures") / "ct_and_torque_versus_tsr.png"
    plot_ct_torque_vs_tsr(tsr_values, ct_values, torque_values, output_file)

    print(f"Saved Ct/torque vs TSR graph to: {output_file}")


if __name__ == "__main__":
    main()
