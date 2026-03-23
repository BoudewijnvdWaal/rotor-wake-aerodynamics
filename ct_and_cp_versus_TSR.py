"""
Generate total thrust, power, and torque coefficients versus TSR from the BEM model and save the figure.
"""

from pathlib import Path
import contextlib
import io

import matplotlib.pyplot as plt
import numpy as np
import BEM_TG2303 as bem

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


def compute_coefficients_vs_tsr(tsr_values, n_annuli=100):
    """Run BEM for each TSR and return Ct, Cp, and Cq coefficient arrays."""
    ct_values = np.zeros(len(tsr_values))
    cp_values = np.zeros(len(tsr_values))
    cq_values = np.zeros(len(tsr_values))

    for i, tsr in enumerate(tsr_values):
        omega = tsr * U0 / Radius
        r_R, chord_distribution, twist_distribution, _, _ = initialise(n_annuli)

        # Suppress BEM solver output
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
        cq_values[i] = cp / tsr  # Cq = Cp / TSR

    return ct_values, cp_values, cq_values


def plot_coefficients_vs_tsr(tsr_values, ct_values, cp_values, cq_values, output_file):
    """Create and save a three-subplot figure for Ct, Cp, and Cq versus TSR."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Total Thrust Coefficient
    axes[0].plot(
        tsr_values,
        ct_values,
        "o-",
        color="tab:blue",
        linewidth=2,
        markersize=7,
        label="Total Thrust Coefficient",
    )
    axes[0].set_title("Total Thrust Coefficient vs TSR", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Tip-Speed Ratio (TSR)", fontsize=11)
    axes[0].set_ylabel(r"$C_T$", fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Total Power Coefficient
    axes[1].plot(
        tsr_values,
        cp_values,
        "s-",
        color="tab:red",
        linewidth=2,
        markersize=7,
        label="Total Power Coefficient",
    )
    axes[1].set_title("Total Power Coefficient vs TSR", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Tip-Speed Ratio (TSR)", fontsize=11)
    axes[1].set_ylabel(r"$C_P$", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Total Torque Coefficient
    axes[2].plot(
        tsr_values,
        cq_values,
        "^-",
        color="tab:green",
        linewidth=2,
        markersize=7,
        label="Total Torque Coefficient",
    )
    axes[2].set_title("Total Torque Coefficient vs TSR", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Tip-Speed Ratio (TSR)", fontsize=11)
    axes[2].set_ylabel(r"$C_Q$", fontsize=11)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        "Discretisation and Spacing Methods Effects on Rotor Coefficients",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _execute_with_optional_prandtl(tsr, n_annuli=100, use_prandtl=True):
    """Run executeBEM at one TSR with either active or disabled Prandtl correction."""
    omega = tsr * U0 / Radius
    r_R, chord_distribution, twist_distribution, _, _ = initialise(n_annuli)

    original_prandtl_fn = bem.PrandtlTipRootCorrection

    def no_prandtl_correction(*_args, **_kwargs):
        return 1.0, 1.0, 1.0

    try:
        if not use_prandtl:
            bem.PrandtlTipRootCorrection = no_prandtl_correction

        with contextlib.redirect_stdout(io.StringIO()):
            _, _, results, _, _, _ = executeBEM(
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
    finally:
        bem.PrandtlTipRootCorrection = original_prandtl_fn

    return results


def compute_prandtl_influence_data(tsr_values, n_annuli=100):
    """Compute thrust loading distributions with and without Prandtl correction."""
    influence_data = {}

    for tsr in tsr_values:
        results_with = _execute_with_optional_prandtl(tsr, n_annuli=n_annuli, use_prandtl=True)
        results_without = _execute_with_optional_prandtl(tsr, n_annuli=n_annuli, use_prandtl=False)

        influence_data[tsr] = {
            "r_over_R": results_with[:, 2],
            "thrust_with": results_with[:, 3],
            "thrust_without": results_without[:, 3],
        }

    return influence_data


def plot_prandtl_influence_three_tsr(influence_data, output_file):
    """Plot Prandtl correction influence for TSR 6, 8, and 10 in one figure."""
    tsr_values = list(influence_data.keys())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)

    for ax, tsr in zip(axes, tsr_values):
        data = influence_data[tsr]
        ax.plot(
            data["r_over_R"],
            data["thrust_with"],
            color="tab:blue",
            linewidth=1.8,
            label="With Prandtl Correction",
        )
        ax.plot(
            data["r_over_R"],
            data["thrust_without"],
            linestyle="--",
            color="tab:orange",
            linewidth=1.5,
            label="No Correction",
        )

        ax.set_title(f"Influence of Prandtl Tip and Root Correction (TSR {tsr})", fontsize=11)
        ax.set_xlabel("r/R", fontsize=11)
        ax.grid(True, alpha=0.4)

    axes[0].set_ylabel("Thrust Loading (N/m)", fontsize=11)
    axes[1].legend(loc="upper left", fontsize=10)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    tsr_values = np.arange(4.0, 18.5, 0.5)
    ct_values, cp_values, cq_values = compute_coefficients_vs_tsr(tsr_values, n_annuli=100)

    output_file = Path("figures") / "coefficients_versus_tsr.png"
    plot_coefficients_vs_tsr(tsr_values, ct_values, cp_values, cq_values, output_file)

    prandtl_tsr_values = [6, 8, 10]
    prandtl_data = compute_prandtl_influence_data(prandtl_tsr_values, n_annuli=100)
    prandtl_output_file = Path("figures") / "prandtl_influence_tsr_6_8_10.png"
    plot_prandtl_influence_three_tsr(prandtl_data, prandtl_output_file)

    print(f"Saved coefficient vs TSR graph to: {output_file}")
    print(f"Saved Prandtl influence graph to: {prandtl_output_file}")


if __name__ == "__main__":
    main()
