"""
Blade Element Momentum (BEM) Theory Model for a Wind Turbine Rotor
Authors: Thijmen God, Boudewijn van der Waal, Rens van Lierop
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob


# --------- Module 0 : Initialistation ---------

# BLOCK 0.1 : Rotor specs
Radius = 50  # m
blades = 3
RootLocation_R = 0.2
TipLocation_R = 1.0
blade_pitch = -2  # deg

def initialise(N):
    delta_r_R = (TipLocation_R - RootLocation_R) / N
    r_R = np.linspace(RootLocation_R, TipLocation_R, N + 1)
    chord_distribution = 3 * (1 - r_R) + 1
    twist_distribution = 14 * (1 - r_R)

    a = np.zeros(len(r_R) - 1) + 0.3
    aline = np.zeros(len(r_R) - 1)

    return r_R, chord_distribution, twist_distribution, a, aline

r_R, chord_distribution, twist_distribution, a, aline = initialise(100)

# BLOCK 0.4 : Operational specs
U0 = 10
TSR = [6, 8, 10]
rotor_yaw = 0
Omega = [TSR[i] * U0 / Radius for i in range(len(TSR))]


# --------- Module 1 : Lift and Drag Coefficients ---------

def load_polar():
    possible_files = [
        "polar DU95W180.xlsx",
        "polar DU95W180 (3).xlsx",
        "/mnt/data/polar DU95W180.xlsx",
        "/mnt/data/polar DU95W180 (3).xlsx"
    ]

    airfoil_file = None
    for f in possible_files:
        if os.path.exists(f):
            airfoil_file = f
            break

    if airfoil_file is None:
        matches = glob.glob("*DU95W180*.xlsx") + glob.glob("/mnt/data/*DU95W180*.xlsx")
        if matches:
            airfoil_file = matches[0]
        else:
            raise FileNotFoundError("DU95W180 polar file not found.")

    data = pd.read_excel(airfoil_file, header=0, names=["alpha", "cl", "cd", "cm"]).dropna()
    data["alpha"] = pd.to_numeric(data["alpha"], errors="coerce")
    data["cl"] = pd.to_numeric(data["cl"], errors="coerce")
    data["cd"] = pd.to_numeric(data["cd"], errors="coerce")
    data = data.dropna()

    polar_alpha = data["alpha"].to_numpy(dtype=float)
    polar_cl = data["cl"].to_numpy(dtype=float)
    polar_cd = data["cd"].to_numpy(dtype=float)

    return polar_alpha, polar_cl, polar_cd

polar_alpha, polar_cl, polar_cd = load_polar()


# --------- Module 2 : Normal and Tangential Forces ---------

def CTfunction(a, glauert_correction=False):
    CT = 4 * a * (1 - a)
    if glauert_correction:
        CT1 = 1.816
        a1 = 1 - np.sqrt(CT1) / 2
        CT = np.where(a > a1, CT1 - 4 * (np.sqrt(CT1) - 1) * (1 - a), CT)
    return CT

def ainduction(CT, glauert_correction=True):
    CT = np.asarray(CT, dtype=float)

    if not glauert_correction:
        CT = np.clip(CT, None, 1.0)
        return 0.5 - 0.5 * np.sqrt(1 - CT)

    a = np.zeros(np.shape(CT))
    CT1 = 1.816
    CT2 = 2 * np.sqrt(CT1) - CT1

    mask_high = CT >= CT2
    mask_low = CT < CT2

    a[mask_high] = 1 + (CT[mask_high] - CT1) / (4 * (np.sqrt(CT1) - 1))
    CT_low = np.clip(CT[mask_low], None, 1.0)
    a[mask_low] = 0.5 - 0.5 * np.sqrt(1 - CT_low)

    return a

def PrandtlTipRootCorrection(r_R, rootradius_R, tipradius_R, TSR, NBlades, axial_induction):
    denom = max(1 - axial_induction, 1e-6)

    temp_tip = -NBlades / 2 * (tipradius_R - r_R) / r_R * np.sqrt(1 + ((TSR * r_R) ** 2) / (denom ** 2))
    Ftip = np.array(2 / np.pi * np.arccos(np.exp(temp_tip)))
    Ftip[np.isnan(Ftip)] = 0

    temp_root = -NBlades / 2 * (r_R - rootradius_R) / r_R * np.sqrt(1 + ((TSR * r_R) ** 2) / (denom ** 2))
    Froot = np.array(2 / np.pi * np.arccos(np.exp(temp_root)))
    Froot[np.isnan(Froot)] = 0

    return Froot * Ftip, Ftip, Froot

def loadBladeElement(vnorm, vtan, r_R, chord, twist, blade_pitch, polar_alpha, polar_cl, polar_cd):
    vmag2 = vnorm**2 + vtan**2
    inflowangle = np.arctan2(vnorm, vtan)
    inflowangle_deg = inflowangle * 180 / np.pi

    # Correct wind-turbine angle of attack
    alpha = inflowangle_deg - (twist + blade_pitch)

    cl = np.interp(alpha, polar_alpha, polar_cl)
    cd = np.interp(alpha, polar_alpha, polar_cd)

    lift = 0.5 * vmag2 * cl * chord
    drag = 0.5 * vmag2 * cd * chord

    fnorm = lift * np.cos(inflowangle) + drag * np.sin(inflowangle)
    ftan = lift * np.sin(inflowangle) - drag * np.cos(inflowangle)
    gamma = 0.5 * np.sqrt(vmag2) * cl * chord

    return fnorm, ftan, gamma, alpha, inflowangle_deg, cl, cd


# --------- Module 3 : Streamtube Model ---------

def solveStreamtube(Uinf, r1_R, r2_R, rootradius_R, tipradius_R, Omega, Radius, NBlades,
                    chord, twist, polar_alpha, polar_cl, polar_cd,
                    use_glauert=True, use_prandtl=True):

    Area = np.pi * ((r2_R * Radius) ** 2 - (r1_R * Radius) ** 2)
    r_R = (r1_R + r2_R) / 2

    a = 0.3
    aline = 0.0
    anew = 1.0

    iteration = 0
    Erroriterations = 1e-5
    max_iterations = 1000
    iteration_history = []

    while np.abs(a - anew) > Erroriterations and iteration < max_iterations:
        iteration += 1

        Urotor = Uinf * (1 - a)
        Utan = (1 + aline) * Omega * r_R * Radius

        fnorm, ftan, gamma, alpha, phi, cl, cd = loadBladeElement(
            Urotor, Utan, r_R, chord, twist, blade_pitch, polar_alpha, polar_cl, polar_cd
        )

        load3Daxial = fnorm * Radius * (r2_R - r1_R) * NBlades
        CT_annulus = load3Daxial / (0.5 * Area * Uinf**2)

        anew = float(np.atleast_1d(ainduction(CT_annulus, glauert_correction=use_glauert))[0])

        Prandtl = 1.0
        if use_prandtl:
            Prandtl, Prandtltip, Prandtlroot = PrandtlTipRootCorrection(
                r_R, rootradius_R, tipradius_R, Omega * Radius / Uinf, NBlades, anew
            )
            if Prandtl < 1e-4:
                Prandtl = 1e-4
            anew = anew / Prandtl

        a = 0.75 * a + 0.25 * anew

        aline_new = ftan * NBlades / (4 * np.pi * Uinf * (1 - a) * Omega * (r_R * Radius) ** 2)

        if use_prandtl:
            aline_new = aline_new / Prandtl

        aline = 0.75 * aline + 0.25 * aline_new

        a = np.clip(a, -0.2, 0.95)
        aline = np.clip(aline, -1.0, 1.0)

        iteration_history.append(a)

    return [a, aline, r_R, fnorm, ftan, gamma, alpha, phi, cl, cd, iteration_history]


# --------- Module 4 : BEM executor ---------

def executeBEM(Uinf, TSR, RootLocation_R, TipLocation_R, Omega, Radius, NBlades,
               chord_distribution, twist_distribution, polar_alpha, polar_cl, polar_cd,
               use_glauert=True, use_prandtl=True):

    # columns:
    # 0 a, 1 aline, 2 r/R, 3 fnorm, 4 ftan, 5 gamma, 6 alpha, 7 phi, 8 cl, 9 cd, 10 Ct_local, 11 Cn_local, 12 Cq_local
    results = np.zeros([len(r_R) - 1, 13])
    iteration_lengths = []

    for i in range(len(r_R) - 1):
        r_mid = 0.5 * (r_R[i] + r_R[i + 1])
        chord = np.interp(r_mid, r_R, chord_distribution)
        twist = np.interp(r_mid, r_R, twist_distribution)

        out = solveStreamtube(
            Uinf, r_R[i], r_R[i + 1], RootLocation_R, TipLocation_R,
            Omega, Radius, NBlades, chord, twist, polar_alpha, polar_cl, polar_cd,
            use_glauert=use_glauert, use_prandtl=use_prandtl
        )

        a_i, aline_i, r_i, fnorm, ftan, gamma, alpha, phi, cl, cd, hist = out
        iteration_lengths.append(len(hist))

        Ct_local = fnorm / (0.5 * Uinf**2 * Radius)
        Cq_local = ftan / (0.5 * Uinf**2 * Radius)
        Cn_local = fnorm / (0.5 * Uinf**2 * np.interp(r_i, r_R, chord_distribution))

        results[i, :] = [a_i, aline_i, r_i, fnorm, ftan, gamma, alpha, phi, cl, cd, Ct_local, Cn_local, Cq_local]

    dr = (r_R[1:] - r_R[:-1]) * Radius
    CT = np.sum(dr * results[:, 3] * blades / (0.5 * Uinf**2 * np.pi * Radius**2))
    CP = np.sum(dr * results[:, 4] * results[:, 2] * blades * Radius * Omega / (0.5 * Uinf**3 * np.pi * Radius**2))

    rho = 1.225
    Thrust = 0.5 * rho * Uinf**2 * np.pi * Radius**2 * CT
    Power = 0.5 * rho * Uinf**3 * np.pi * Radius**2 * CP
    Torque = Power / Omega

    return CT, CP, results, Thrust, Torque, iteration_lengths


# --------- Module 5 : Plotting ---------

def plot_spanwise_baseline_for_tsr(tsr_value, results, use_prandtl, use_glauert):
    label = f"TSR = {tsr_value}, Prandtl={use_prandtl}, Glauert={use_glauert}"

    # alpha and inflow
    plt.figure(figsize=(12, 6))
    plt.title("Spanwise distribution of angle of attack and inflow angle\n" + label)
    plt.plot(results[:, 2], results[:, 6], 'b-', label='Angle of attack α [deg]')
    plt.plot(results[:, 2], results[:, 7], 'r--', label='Inflow angle φ [deg]')
    plt.xlabel('r/R')
    plt.ylabel('Angle [deg]')
    plt.grid()
    plt.legend()
    plt.show()

    # a and a'
    plt.figure(figsize=(12, 6))
    plt.title("Spanwise distribution of axial and azimuthal induction\n" + label)
    plt.plot(results[:, 2], results[:, 0], 'r-', label='a')
    plt.plot(results[:, 2], results[:, 1], 'g--', label="a'")
    plt.xlabel('r/R')
    plt.ylabel('Induction factor [-]')
    plt.grid()
    plt.legend()
    plt.show()

    # dimensional loads
    plt.figure(figsize=(12, 6))
    plt.title("Spanwise distribution of thrust and azimuthal loading\n" + label)
    plt.plot(results[:, 2], results[:, 3], 'b-', label='Normal load [N/m]')
    plt.plot(results[:, 2], results[:, 4], 'r--', label='Tangential load [N/m]')
    plt.xlabel('r/R')
    plt.ylabel('Load [N/m]')
    plt.grid()
    plt.legend()
    plt.show()

    # Ct / Cn / Cq
    plt.figure(figsize=(12, 6))
    plt.title("Spanwise distribution of Ct, Cn and Cq\n" + label)
    plt.plot(results[:, 2], results[:, 10], 'b-', label='Ct')
    plt.plot(results[:, 2], results[:, 11], 'g--', label='Cn')
    plt.plot(results[:, 2], results[:, 12], 'r-.', label='Cq')
    plt.xlabel('r/R')
    plt.ylabel('Coefficient [-]')
    plt.grid()
    plt.legend()
    plt.show()


def plot_tip_correction_comparison_for_tsr(tsr_value, results_with, results_without):
    # alpha / inflow
    plt.figure(figsize=(12, 6))
    plt.title(f"Influence of corrections on α and φ, TSR = {tsr_value}")
    plt.plot(results_with[:, 2], results_with[:, 6], 'b-', label='α with corrections')
    plt.plot(results_without[:, 2], results_without[:, 6], 'b--', label='α without corrections')
    plt.plot(results_with[:, 2], results_with[:, 7], 'r-', label='φ with corrections')
    plt.plot(results_without[:, 2], results_without[:, 7], 'r--', label='φ without corrections')
    plt.xlabel('r/R')
    plt.ylabel('Angle [deg]')
    plt.grid()
    plt.legend()
    plt.show()

    # a / a'
    plt.figure(figsize=(12, 6))
    plt.title(f"Influence of corrections on a and a', TSR = {tsr_value}")
    plt.plot(results_with[:, 2], results_with[:, 0], 'b-', label='a with corrections')
    plt.plot(results_without[:, 2], results_without[:, 0], 'b--', label='a without corrections')
    plt.plot(results_with[:, 2], results_with[:, 1], 'r-', label="a' with corrections")
    plt.plot(results_without[:, 2], results_without[:, 1], 'r--', label="a' without corrections")
    plt.xlabel('r/R')
    plt.ylabel('Induction factor [-]')
    plt.grid()
    plt.legend()
    plt.show()

    # loads
    plt.figure(figsize=(12, 6))
    plt.title(f"Influence of corrections on spanwise loading, TSR = {tsr_value}")
    plt.plot(results_with[:, 2], results_with[:, 3], 'b-', label='Fn with corrections')
    plt.plot(results_without[:, 2], results_without[:, 3], 'b--', label='Fn without corrections')
    plt.plot(results_with[:, 2], results_with[:, 4], 'r-', label='Ft with corrections')
    plt.plot(results_without[:, 2], results_without[:, 4], 'r--', label='Ft without corrections')
    plt.xlabel('r/R')
    plt.ylabel('Load [N/m]')
    plt.grid()
    plt.legend()
    plt.show()

    # Ct / Cn / Cq
    plt.figure(figsize=(12, 6))
    plt.title(f"Influence of corrections on Ct, Cn and Cq, TSR = {tsr_value}")
    plt.plot(results_with[:, 2], results_with[:, 10], 'b-', label='Ct with corrections')
    plt.plot(results_without[:, 2], results_without[:, 10], 'b--', label='Ct without corrections')
    plt.plot(results_with[:, 2], results_with[:, 11], 'g-', label='Cn with corrections')
    plt.plot(results_without[:, 2], results_without[:, 11], 'g--', label='Cn without corrections')
    plt.plot(results_with[:, 2], results_with[:, 12], 'r-', label='Cq with corrections')
    plt.plot(results_without[:, 2], results_without[:, 12], 'r--', label='Cq without corrections')
    plt.xlabel('r/R')
    plt.ylabel('Coefficient [-]')
    plt.grid()
    plt.legend()
    plt.show()


def compare_corrections():
    CT_with = np.zeros(len(TSR))
    CP_with = np.zeros(len(TSR))
    Thrust_with = np.zeros(len(TSR))
    Torque_with = np.zeros(len(TSR))

    CT_without = np.zeros(len(TSR))
    CP_without = np.zeros(len(TSR))
    Thrust_without = np.zeros(len(TSR))
    Torque_without = np.zeros(len(TSR))

    all_results_with = []
    all_results_without = []

    for j in range(len(TSR)):
        # WITH corrections
        CT, CP, results_with, Thrust, Torque, iter_len_with = executeBEM(
            U0, TSR[j], RootLocation_R, TipLocation_R, Omega[j], Radius, blades,
            chord_distribution, twist_distribution, polar_alpha, polar_cl, polar_cd,
            use_glauert=True, use_prandtl=True
        )

        CT_with[j] = CT
        CP_with[j] = CP
        Thrust_with[j] = Thrust
        Torque_with[j] = Torque
        all_results_with.append(results_with)

        # WITHOUT corrections
        CT, CP, results_without, Thrust, Torque, iter_len_without = executeBEM(
            U0, TSR[j], RootLocation_R, TipLocation_R, Omega[j], Radius, blades,
            chord_distribution, twist_distribution, polar_alpha, polar_cl, polar_cd,
            use_glauert=False, use_prandtl=False
        )

        CT_without[j] = CT
        CP_without[j] = CP
        Thrust_without[j] = Thrust
        Torque_without[j] = Torque
        all_results_without.append(results_without)

        print(f"\nTSR = {TSR[j]}")
        print(f"With corrections    : CT = {CT_with[j]:.4f}, CP = {CP_with[j]:.4f}, Thrust = {Thrust_with[j]:.2f} N, Torque = {Torque_with[j]:.2f} Nm")
        print(f"Without corrections : CT = {CT_without[j]:.4f}, CP = {CP_without[j]:.4f}, Thrust = {Thrust_without[j]:.2f} N, Torque = {Torque_without[j]:.2f} Nm")

        # Baseline plots for this TSR
        plot_spanwise_baseline_for_tsr(TSR[j], results_with, use_prandtl=True, use_glauert=True)

        # Comparison plots for this TSR
        plot_tip_correction_comparison_for_tsr(TSR[j], results_with, results_without)

    # Summary plot: total thrust and torque versus TSR
    plt.figure(figsize=(12, 6))
    plt.title("Total thrust and torque versus TSR")
    plt.plot(TSR, Thrust_with, 'bo-', label='Thrust with corrections')
    plt.plot(TSR, Thrust_without, 'bs--', label='Thrust without corrections')
    plt.plot(TSR, Torque_with, 'ro-', label='Torque with corrections')
    plt.plot(TSR, Torque_without, 'rs--', label='Torque without corrections')
    plt.xlabel('Tip speed ratio λ')
    plt.ylabel('Load')
    plt.grid()
    plt.legend()
    plt.show()

    # Summary plot: CT and CP versus TSR
    plt.figure(figsize=(12, 6))
    plt.title("CT and CP versus TSR")
    plt.plot(TSR, CT_with, 'bo-', label='CT with corrections')
    plt.plot(TSR, CT_without, 'bs--', label='CT without corrections')
    plt.plot(TSR, CP_with, 'ro-', label='CP with corrections')
    plt.plot(TSR, CP_without, 'rs--', label='CP without corrections')
    plt.xlabel('Tip speed ratio λ')
    plt.ylabel('Coefficient [-]')
    plt.grid()
    plt.legend()
    plt.show()


def main():
    compare_corrections()

if __name__ == "__main__":
    main()