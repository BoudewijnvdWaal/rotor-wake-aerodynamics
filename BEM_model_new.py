import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# --------- Module 0 : Initialisation ---------

Radius = 50.0
blades = 3
RootLocation_R = 0.2
TipLocation_R = 1.0
blade_pitch = -2.0  # degrees

delta_r_R = 0.01
r_R = np.arange(RootLocation_R, TipLocation_R + delta_r_R, delta_r_R)

chord_distribution = 3 * (1 - r_R) + 1                 # m
twist_distribution = 14 * (1 - r_R) + blade_pitch      # include collective pitch here

U0 = 10.0
TSR_list = [6, 8, 10]
Omega_list = [tsr * U0 / Radius for tsr in TSR_list]

glauert_correction = True
prandtl_correction = True

rho = 1.225  # if your lecturer assumes incompressible sea-level standard air, use this


# --------- Module 1 : Airfoil polar ---------

airfoil = "polar DU95W180.xlsx"
data = pd.read_excel(airfoil, header=0, names=["alpha", "cl", "cd", "cm"]).dropna()
data = data.drop(data.index[0])

polar_alpha = data["alpha"].astype(float).to_numpy()
polar_cl = data["cl"].astype(float).to_numpy()
polar_cd = data["cd"].astype(float).to_numpy()


# --------- Module 2 : Aerodynamics ---------

def ainduction(CT):
    a = np.zeros(np.shape(CT))
    CT1 = 1.816
    CT2 = 2 * np.sqrt(CT1) - CT1
    a = np.asarray(CT, dtype=float)

    out = np.zeros_like(a)
    high = a >= CT2
    low = ~high

    out[high] = 1 + (a[high] - CT1) / (4 * (np.sqrt(CT1) - 1))
    out[low] = 0.5 - 0.5 * np.sqrt(1 - a[low])
    return out


def PrandtlTipRootCorrection(r_R, rootradius_R, tipradius_R, TSR, NBlades, axial_induction):
    axial_induction = np.clip(axial_induction, -0.95, 0.95)

    temp_tip = -NBlades / 2 * (tipradius_R - r_R) / r_R * np.sqrt(
        1 + ((TSR * r_R) ** 2) / ((1 - axial_induction) ** 2)
    )
    Ftip = np.array(2 / np.pi * np.arccos(np.exp(temp_tip)))
    Ftip[np.isnan(Ftip)] = 0.0

    temp_root = NBlades / 2 * (rootradius_R - r_R) / r_R * np.sqrt(
        1 + ((TSR * r_R) ** 2) / ((1 - axial_induction) ** 2)
    )
    Froot = np.array(2 / np.pi * np.arccos(np.exp(temp_root)))
    Froot[np.isnan(Froot)] = 0.0

    F = Froot * Ftip
    return F, Ftip, Froot


def loadBladeElement(vnorm, vtan, chord, theta_deg, polar_alpha, polar_cl, polar_cd):
    vrel2 = vnorm**2 + vtan**2
    phi = np.arctan2(vnorm, vtan)                  # inflow angle [rad]
    phi_deg = np.degrees(phi)

    alpha_deg = theta_deg - phi_deg                # standard convention
    cl = np.interp(alpha_deg, polar_alpha, polar_cl)
    cd = np.interp(alpha_deg, polar_alpha, polar_cd)

    L = 0.5 * rho * vrel2 * cl * chord             # N/m
    D = 0.5 * rho * vrel2 * cd * chord             # N/m

    fnorm = L * np.cos(phi) + D * np.sin(phi)      # axial/normal force per m
    ftan = L * np.sin(phi) - D * np.cos(phi)       # tangential force per m

    return {
        "phi_rad": phi,
        "phi_deg": phi_deg,
        "alpha_deg": alpha_deg,
        "cl": cl,
        "cd": cd,
        "fnorm": fnorm,
        "ftan": ftan,
        "vrel": np.sqrt(vrel2),
    }


# --------- Module 3 : Streamtube solver ---------

def solveStreamtube(
    Uinf, r1_R, r2_R, rootradius_R, tipradius_R, Omega, Radius, NBlades,
    chord, theta_deg, polar_alpha, polar_cl, polar_cd,
    use_prandtl=True, max_iter=500, tol=1e-5
):
    area = np.pi * ((r2_R * Radius) ** 2 - (r1_R * Radius) ** 2)
    rmid_R = 0.5 * (r1_R + r2_R)
    rmid = rmid_R * Radius
    dr = (r2_R - r1_R) * Radius

    a = 0.0
    aline = 0.0
    history = []

    for it in range(max_iter):
        Uax = Uinf * (1 - a)
        Utan = Omega * rmid * (1 + aline)

        aero = loadBladeElement(Uax, Utan, chord, theta_deg, polar_alpha, polar_cl, polar_cd)

        dT = aero["fnorm"] * dr * NBlades
        dQ = aero["ftan"] * rmid * dr * NBlades

        CT_annulus = dT / (0.5 * rho * Uinf**2 * area)
        anew = ainduction(np.array([CT_annulus]))[0]

        F = 1.0
        Ftip = 1.0
        Froot = 1.0

        if use_prandtl:
            F, Ftip, Froot = PrandtlTipRootCorrection(
                rmid_R, rootradius_R, tipradius_R, Omega * Radius / Uinf, NBlades, anew
            )
            F = max(F, 1e-4)
            anew = anew / F

        a_next = 0.75 * a + 0.25 * anew

        aline_new = aero["ftan"] * NBlades / (
            2 * np.pi * rho * Uinf * (1 - a_next) * Omega * 2 * rmid**2
        )
        if use_prandtl:
            aline_new = aline_new / F

        history.append(a_next)

        if abs(a_next - a) < tol:
            a = a_next
            aline = aline_new
            break

        a = a_next
        aline = aline_new

    Cn_local = aero["fnorm"] / (0.5 * rho * Uinf**2 * chord)
    Ct_local = aero["ftan"] / (0.5 * rho * Uinf**2 * chord)
    Cq_local = Ct_local * rmid_R

    return {
        "r_R": rmid_R,
        "r": rmid,
        "dr": dr,
        "a": a,
        "aline": aline,
        "phi_deg": aero["phi_deg"],
        "alpha_deg": aero["alpha_deg"],
        "cl": aero["cl"],
        "cd": aero["cd"],
        "fnorm": aero["fnorm"],
        "ftan": aero["ftan"],
        "Cn": Cn_local,
        "Ct": Ct_local,
        "Cq": Cq_local,
        "F": F,
        "Ftip": Ftip,
        "Froot": Froot,
        "iterations": len(history),
        "a_history": history,
        "dT": dT,
        "dQ": dQ,
        "chord": chord,
        "theta_deg": theta_deg,
    }


# --------- Module 4 : BEM executor ---------

def executeBEM(
    Uinf, TSR, RootLocation_R, TipLocation_R, Omega, Radius, NBlades,
    chord_distribution, twist_distribution, polar_alpha, polar_cl, polar_cd,
    use_prandtl=True
):
    rows = []

    for i in range(len(r_R) - 1):
        rmid_R = 0.5 * (r_R[i] + r_R[i + 1])
        chord = np.interp(rmid_R, r_R, chord_distribution)
        theta_deg = np.interp(rmid_R, r_R, twist_distribution)

        row = solveStreamtube(
            Uinf, r_R[i], r_R[i + 1], RootLocation_R, TipLocation_R,
            Omega, Radius, NBlades, chord, theta_deg,
            polar_alpha, polar_cl, polar_cd,
            use_prandtl=use_prandtl
        )
        rows.append(row)

    df = pd.DataFrame(rows)

    A = np.pi * Radius**2
    T_total = df["dT"].sum()
    Q_total = df["dQ"].sum()
    P_total = Q_total * Omega

    CT = T_total / (0.5 * rho * Uinf**2 * A)
    CP = P_total / (0.5 * rho * Uinf**3 * A)

    summary = {
        "TSR": TSR,
        "Omega": Omega,
        "T_total": T_total,
        "Q_total": Q_total,
        "P_total": P_total,
        "CT": CT,
        "CP": CP,
        "df": df,
    }
    return summary


# --------- Module 5 : Plotting ---------

def plot_spanwise_comparison(result_with, result_without, tsr):
    df_w = result_with["df"]
    df_wo = result_without["df"]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(df_w["r_R"], df_w["alpha_deg"], label="with tip correction")
    axs[0, 0].plot(df_wo["r_R"], df_wo["alpha_deg"], "--", label="without tip correction")
    axs[0, 0].set_xlabel("r/R")
    axs[0, 0].set_ylabel("alpha [deg]")
    axs[0, 0].grid()
    axs[0, 0].legend()

    axs[0, 1].plot(df_w["r_R"], df_w["a"], label="with tip correction")
    axs[0, 1].plot(df_wo["r_R"], df_wo["a"], "--", label="without tip correction")
    axs[0, 1].set_xlabel("r/R")
    axs[0, 1].set_ylabel("a [-]")
    axs[0, 1].grid()
    axs[0, 1].legend()

    axs[1, 0].plot(df_w["r_R"], df_w["fnorm"], label="with tip correction")
    axs[1, 0].plot(df_wo["r_R"], df_wo["fnorm"], "--", label="without tip correction")
    axs[1, 0].set_xlabel("r/R")
    axs[1, 0].set_ylabel("normal load [N/m]")
    axs[1, 0].grid()
    axs[1, 0].legend()

    axs[1, 1].plot(df_w["r_R"], df_w["ftan"], label="with tip correction")
    axs[1, 1].plot(df_wo["r_R"], df_wo["ftan"], "--", label="without tip correction")
    axs[1, 1].set_xlabel("r/R")
    axs[1, 1].set_ylabel("tangential load [N/m]")
    axs[1, 1].grid()
    axs[1, 1].legend()

    fig.suptitle(f"Influence of tip/root correction at TSR = {tsr}")
    plt.tight_layout()
    plt.show()


def plot_prandtl_factor(result, tsr):
    df = result["df"]
    plt.figure(figsize=(8, 5))
    plt.plot(df["r_R"], df["F"], label="Combined F")
    plt.plot(df["r_R"], df["Ftip"], label="Tip factor")
    plt.plot(df["r_R"], df["Froot"], label="Root factor")
    plt.xlabel("r/R")
    plt.ylabel("Prandtl factor [-]")
    plt.title(f"Prandtl correction factors at TSR = {tsr}")
    plt.grid()
    plt.legend()
    plt.show()


def plot_total_performance(results_with, results_without):
    tsr_vals = [r["TSR"] for r in results_with]
    ct_with = [r["CT"] for r in results_with]
    cp_with = [r["CP"] for r in results_with]
    ct_without = [r["CT"] for r in results_without]
    cp_without = [r["CP"] for r in results_without]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(tsr_vals, ct_with, "o-", label="with tip correction")
    axs[0].plot(tsr_vals, ct_without, "s--", label="without tip correction")
    axs[0].set_xlabel("TSR")
    axs[0].set_ylabel("CT")
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(tsr_vals, cp_with, "o-", label="with tip correction")
    axs[1].plot(tsr_vals, cp_without, "s--", label="without tip correction")
    axs[1].set_xlabel("TSR")
    axs[1].set_ylabel("CP")
    axs[1].grid()
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def main():
    results_with = []
    results_without = []

    for tsr, omega in zip(TSR_list, Omega_list):
        print(f"Running TSR = {tsr}")

        res_with = executeBEM(
            U0, tsr, RootLocation_R, TipLocation_R, omega, Radius, blades,
            chord_distribution, twist_distribution,
            polar_alpha, polar_cl, polar_cd,
            use_prandtl=True
        )

        res_without = executeBEM(
            U0, tsr, RootLocation_R, TipLocation_R, omega, Radius, blades,
            chord_distribution, twist_distribution,
            polar_alpha, polar_cl, polar_cd,
            use_prandtl=False
        )

        results_with.append(res_with)
        results_without.append(res_without)

        print(f"TSR = {tsr}")
        print(f"  WITH Prandtl:    CT = {res_with['CT']:.4f}, CP = {res_with['CP']:.4f}")
        print(f"  WITHOUT Prandtl: CT = {res_without['CT']:.4f}, CP = {res_without['CP']:.4f}")

        plot_prandtl_factor(res_with, tsr)
        plot_spanwise_comparison(res_with, res_without, tsr)

    plot_total_performance(results_with, results_without)


if __name__ == "__main__":
    main()