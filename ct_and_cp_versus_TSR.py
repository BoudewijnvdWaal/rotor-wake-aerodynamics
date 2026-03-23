"""
Generate Ct and Cp versus TSR from the BEM model and save the figure.
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


def compute_ct_cp_vs_tsr(tsr_values, n_annuli=100):
    """Run BEM for each TSR and return Ct and Cp arrays."""
    ct_values = np.zeros(len(tsr_values))
    cp_values = np.zeros(len(tsr_values))

    for i, tsr in enumerate(tsr_values):
        omega = tsr * U0 / Radius
        r_R, chord_distribution, twist_distribution, _, _ = initialise(n_annuli)

        # Keep this script focused on the final Ct/Cp graph only.
        with contextlib.redirect_stdout(io.StringIO()):
            ct, cp, _, _, _, _ = executeBEM(
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
        cp_values[i] = cp

    return ct_values, cp_values


def plot_ct_cp_vs_tsr(tsr_values, ct_values, cp_values, output_file):
    """Create and save a dual-axis plot for Cp and Ct versus TSR."""
    fig, ax_cp = plt.subplots(figsize=(10, 6))
    ax_ct = ax_cp.twinx()

    line_cp, = ax_cp.plot(
        tsr_values,
        cp_values,
        "o-",
        color="tab:blue",
        linewidth=1.8,
        markersize=6,
        label="Power Coefficient (C_p)",
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

    ax_cp.set_title("Performance vs TSR")
    ax_cp.set_xlabel("Tip-Speed Ratio (TSR)")
    ax_cp.set_ylabel("Power Coefficient (C_p)", color="tab:blue")
    ax_ct.set_ylabel("Thrust Coefficient (C_t)", color="tab:red")

    ax_cp.tick_params(axis="y", labelcolor="tab:blue")
    ax_ct.tick_params(axis="y", labelcolor="tab:red")
    ax_cp.grid(True)

    ax_cp.legend([line_cp, line_ct], [line_cp.get_label(), line_ct.get_label()], loc="best")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    tsr_values = np.arange(4.0, 12.5, 0.5)
    ct_values, cp_values = compute_ct_cp_vs_tsr(tsr_values, n_annuli=100)

    output_file = Path("figures") / "ct_and_cp_versus_tsr.png"
    plot_ct_cp_vs_tsr(tsr_values, ct_values, cp_values, output_file)

    print(f"Saved Ct/Cp vs TSR graph to: {output_file}")


if __name__ == "__main__":
    main()
