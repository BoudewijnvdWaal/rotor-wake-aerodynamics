"""
Generate total thrust and total torque versus TSR from the BEM model and save the figure.
"""

from pathlib import Path
import contextlib
import io

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

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


def compute_thrust_torque_vs_tsr(tsr_values, n_annuli=100):
    """Run BEM for each TSR and return total thrust and torque arrays."""
    thrust_values = np.zeros(len(tsr_values))
    torque_values = np.zeros(len(tsr_values))

    for i, tsr in enumerate(tsr_values):
        omega = tsr * U0 / Radius
        r_R, chord_distribution, twist_distribution, _, _ = initialise(n_annuli)

        # Keep this script focused on the final Ct/Cp graph only.
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, _, thrust, torque, _ = executeBEM(
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

        thrust_values[i] = thrust
        torque_values[i] = torque

    return thrust_values, torque_values


def plot_thrust_torque_vs_tsr(tsr_values, thrust_values, torque_values, output_file):
    """Create and save a dual-axis plot for total thrust and torque versus TSR."""
    fig, ax_torque = plt.subplots(figsize=(10, 6))
    ax_thrust = ax_torque.twinx()

    line_torque, = ax_torque.plot(
        tsr_values,
        torque_values,
        "o-",
        color="tab:blue",
        linewidth=1.8,
        markersize=6,
        label="Total Torque (Nm)",
    )
    line_thrust, = ax_thrust.plot(
        tsr_values,
        thrust_values,
        "x--",
        color="tab:red",
        linewidth=1.8,
        markersize=6,
        label="Total Thrust (N)",
    )

    ax_torque.set_title("Total Rotor Thrust and Torque vs Tip-Speed Ratio")
    ax_torque.set_xlabel("Tip-Speed Ratio (TSR)")
    ax_torque.set_ylabel("Total Torque (Nm)", color="tab:blue")
    ax_thrust.set_ylabel("Total Thrust (N)", color="tab:red")

    ax_torque.tick_params(axis="y", labelcolor="tab:blue")
    ax_thrust.tick_params(axis="y", labelcolor="tab:red")
    # Use scaled tick labels (manual scientific notation) so the x10^x text can be placed at the bottom.
    ax_torque.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y / 1e6:.1f}"))
    ax_thrust.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y / 1e5:.1f}"))
    ax_torque.grid(True)

    legend = ax_torque.legend(
        [line_torque, line_thrust],
        [line_torque.get_label(), line_thrust.get_label()],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        ncol=1,
        framealpha=1.0,
    )
    legend.set_zorder(10)

    # Manually place scientific notation indicators at the bottom of both y-axes.
    ax_torque.text(
        0.0,
        -0.10,
        r"$\times10^{6}$",
        transform=ax_torque.transAxes,
        color="tab:blue",
        ha="left",
        va="top",
    )
    ax_thrust.text(
        1.0,
        -0.10,
        r"$\times10^{5}$",
        transform=ax_thrust.transAxes,
        color="tab:red",
        ha="right",
        va="top",
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    tsr_values = np.arange(4.0, 12.5, 0.5)
    thrust_values, torque_values = compute_thrust_torque_vs_tsr(tsr_values, n_annuli=100)

    output_file = Path("figures") / "thrust_and_torque_versus_tsr.png"
    plot_thrust_torque_vs_tsr(tsr_values, thrust_values, torque_values, output_file)

    print(f"Saved thrust/torque vs TSR graph to: {output_file}")


if __name__ == "__main__":
    main()
