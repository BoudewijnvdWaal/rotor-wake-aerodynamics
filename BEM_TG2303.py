"""
Blade Element Momentum (BEM) Theory Model for a Wind Turbine Rotor
Authors: Thijmen God, Boudewijn van der Waal, Rens van Lierop
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import shutil


# --------- Module 0 : Initialistation ---------


# BLOCK 0.1 : Rotor specs
Radius = 50 # m
blades = 3 # number of blades
RootLocation_R = 0.2 # m, distance from center where blades start
TipLocation_R = 1.0 # m, distance from center where blades end
blade_pitch = -2 # degrees, pitch angle at the root of the blade
visualise = True # whether to visualise results of BEM solver
FIGURES_DIR = Path("figures")


def prepare_output_directory(base_dir):
    """Recreate clean figures directory for a fresh run."""
    base_dir.mkdir(exist_ok=True)
    for item in base_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def tsr_folder_name(tsr):
    """Return folder suffix that matches requested TSR naming."""
    tsr_value = int(tsr) if float(tsr).is_integer() else tsr
    return f"TSR{tsr_value}"


def save_figure(fig, file_path):
    """Save figure to disk, creating parent folders when needed."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path, dpi=300, bbox_inches="tight")


def save_airfoil_polars(output_dir):
    """Save general airfoil polar plots in the root figures directory."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(polar_alpha, polar_cl, "b-")
    axs[0].set_xlim([-30, 30])
    axs[0].set_xlabel(r"$\alpha$")
    axs[0].set_ylabel(r"$C_l$")
    axs[0].set_title("Lift Coefficient vs Angle of Attack")
    axs[0].grid()

    axs[1].plot(polar_cd, polar_cl, "r-")
    axs[1].set_xlim([0, 0.1])
    axs[1].set_xlabel(r"$C_d$")
    axs[1].set_ylabel(r"$C_l$")
    axs[1].set_title("Lift Coefficient vs Drag Coefficient")
    axs[1].grid()

    save_figure(fig, output_dir / "airfoil_polars.png")


def save_prandtl_distribution(r_over_R, axial_induction, tsr, output_dir):
    """Save Prandtl total, tip and root factors over radial position."""
    # Avoid singular behavior when (1-a) approaches zero in the correction formula.
    axial_safe = np.clip(axial_induction, -0.95, 0.95)
    prandtl_total, prandtl_tip, prandtl_root = PrandtlTipRootCorrection(
        r_over_R,
        RootLocation_R,
        TipLocation_R,
        tsr,
        blades,
        axial_safe,
    )

    fig = plt.figure(figsize=(12, 6))
    plt.plot(r_over_R, prandtl_total, "r-", label="Prandtl total")
    plt.plot(r_over_R, prandtl_tip, "g--", label="Prandtl tip")
    plt.plot(r_over_R, prandtl_root, "b-.", label="Prandtl root")
    plt.xlabel("r/R")
    plt.ylabel("Correction factor")
    plt.title(f"Prandtl correction factors vs radial position (TSR = {tsr})")
    plt.grid()
    plt.legend()
    save_figure(fig, output_dir / f"{tsr_folder_name(tsr)}_prandtl_factors.png")

def initialise(N):
    # BLOCK 0.2 : Section streamtubes
    delta_r_R = 1/N # non-dimensioned width of blade element, in fraction of rotor radius
    #delta_r_R = 0.01 # non-dimensioned width of blade element, in fraction of rotor radius
    r_R = np.arange(RootLocation_R, TipLocation_R + delta_r_R, delta_r_R) # non-dimensioned radial position of blade element, in fraction of rotor radius
    chord_distribution = 3*(1-r_R) + 1 # m, chord distribution along the blade, linearly decreasing from 4 m at the root to 1 m at the tip
    twist_distribution = 14*(1-r_R) # degrees, twist distribution along the blade,

    # BLOCK 0.5 : Prescribe first estimate induction factor for all streamtubes
    a = np.zeros(np.shape(r_R))+0.3 # axial induction factor, initial estimate
    aline = np.zeros(np.shape(r_R)) # tangential induction factor, initial estimate

    return r_R, chord_distribution, twist_distribution, a, aline

# BLOCK 0.4 : Operational specs
U0 = 10 # m/s, free stream velocity
TSR = [6, 8, 10] # Tip Speed Ratios to analyze
rotor_yaw = 0 # degrees, yaw angle of the rotor
Omega = [TSR[i]*U0/Radius for i in range(len(TSR))] # rotational velocity of the rotor, calculated from TSR and free stream velocity
glauert_correction = True # whether to apply Glauert's correction for heavily loaded rotors
prandtl_correction = True # whether to apply Prandtl's tip loss correction

# --------- Module 1 : Lift and Drag Coefficients ---------

# BLOCK 1.1 : Import airfoil data and polar
airfoil = 'polar DU95W180.xlsx'
data=pd.read_excel(airfoil, header=0,
                    names = ["alpha", "cl", "cd", "cm"]).dropna()
data = data.drop(data.index[0]) # drop first row with header info
polar_alpha = data['alpha'][:].astype(float)
polar_cl = data['cl'][:].astype(float)
polar_cd = data['cd'][:].astype(float)


# --------- Module 2 : Normal and Tangential Forces ---------

# BLOCK 2.3 : Calculate thrust coefficient as a function of induction factor, with option to include Glauert's correction for heavily loaded rotors, and calculate axial induction as a function of thrust coefficient, including Glauert's correction
def CTfunction(a, glauert_correction=False):
    """
    This function calculates the thrust coefficient as a function of induction factor 'a'
    'glauert' defines if the Glauert correction for heavily loaded rotors should be used; default value is false
    """
    CT = np.zeros(np.shape(a))
    CT = 4*a*(1-a)  
    if glauert_correction:
        CT1=1.816
        a1=1-np.sqrt(CT1)/2
        CT[a>a1] = CT1-4*(np.sqrt(CT1)-1)*(1-a[a>a1])

    return CT

# BLOCK 2.4 : Calculate axial induction as a function of thrust coefficient, including option to include Glauert's correction for heavily loaded rotors
def ainduction(CT):
    """
    This function calculates the induction factor 'a' as a function of thrust coefficient CT 
    including Glauert's correction
    """
    a = np.zeros(np.shape(CT))
    CT1=1.816
    CT2=2*np.sqrt(CT1)-CT1
    a[CT>=CT2] = 1 + (CT[CT>=CT2]-CT1)/(4*(np.sqrt(CT1)-1))
    a[CT<CT2] = 0.5-0.5*np.sqrt(1-CT[CT<CT2])
    return a

# Block 2.1 : Prandtl's tip and root loss correction
def PrandtlTipRootCorrection(r_R, rootradius_R, tipradius_R, TSR, NBlades, axial_induction):
    """
    This function calcualte steh combined tip and root Prandtl correction at agiven radial position 'r_R' (non-dimensioned by rotor radius), 
    given a root and tip radius (also non-dimensioned), a tip speed ratio TSR, the number lf blades NBlades and the axial induction factor
    """
    temp1 = -NBlades/2*(tipradius_R-r_R)/r_R*np.sqrt( 1+ ((TSR*r_R)**2)/((1-axial_induction)**2))
    Ftip = np.array(2/np.pi*np.arccos(np.exp(temp1)))
    Ftip[np.isnan(Ftip)] = 0
    temp1 = NBlades/2*(rootradius_R-r_R)/r_R*np.sqrt( 1+ ((TSR*r_R)**2)/((1-axial_induction)**2))
    Froot = np.array(2/np.pi*np.arccos(np.exp(temp1)))
    Froot[np.isnan(Froot)] = 0
    return Froot*Ftip, Ftip, Froot

# BLOCK 2.2 : Calculate normal and tangential forces in blade element, given the velocity at the blade element and the polar of the airfoil
def loadBladeElement(vnorm, vtan, r_R, chord, twist, polar_alpha, polar_cl, polar_cd):
    """
    calculates the load in the blade element
    """
    vmag2 = vnorm**2 + vtan**2
    inflowangle = np.arctan2(vnorm,vtan)
    inflowangle_deg = inflowangle * 180/np.pi
    alpha = twist + inflowangle_deg
    cl = np.interp(alpha, polar_alpha, polar_cl)
    cd = np.interp(alpha, polar_alpha, polar_cd)
    lift = 0.5*vmag2*cl*chord
    drag = 0.5*vmag2*cd*chord
    fnorm = lift*np.cos(inflowangle)+drag*np.sin(inflowangle)
    ftan = lift*np.sin(inflowangle)-drag*np.cos(inflowangle)
    gamma = 0.5*np.sqrt(vmag2)*cl*chord

    return fnorm , ftan, gamma, alpha, inflowangle_deg


# --------- Module 3 : Streamtube Model ---------

# BLOCK 3.1 : Solve balance of momentum between blade element load and loading in the streamtube, for a given set of input parameters, and return the axial and tangential induction factor, the normal and tangential load and circulation at the blade element
def solveStreamtube(Uinf, r1_R, r2_R, rootradius_R, tipradius_R , Omega, Radius, NBlades, chord, twist, polar_alpha, polar_cl, polar_cd ):
    """
    solve balance of momentum between blade element load and loading in the streamtube
    input variables:
    Uinf - wind speed at infinity
    r1_R,r2_R - edges of blade element, in fraction of Radius ;
    rootradius_R, tipradius_R - location of blade root and tip, in fraction of Radius ;
    Radius is the rotor radius
    Omega -rotational velocity
    NBlades - number of blades in rotor
    """
    Area = np.pi*((r2_R*Radius)**2-(r1_R*Radius)**2) #  area streamtube
    r_R = (r1_R+r2_R)/2 # centroid
    # initiatlize variables
    a = 0.0 # axial induction
    aline = 0.0 # tangential induction factor
    anew = 0.3 # new estimate of axial induction, for iteration process
    
    iteration = 0
    Erroriterations = 0.000001 # error limit for iteration process, in absolute value of induction

    while abs(anew-a) > Erroriterations:
        iteration += 1
        # calculate velocity and loads at blade element
        Urotor = Uinf*(1-a) # axial velocity at rotor
        Utan = (1+aline)*Omega*r_R*Radius # tangential velocity at rotor
        # calculate loads in blade segment in 2D (N/m)
        fnorm, ftan, gamma, alpha, phi = loadBladeElement(Urotor, Utan, r_R, chord, twist, polar_alpha, polar_cl, polar_cd)
        load3Daxial =fnorm*Radius*(r2_R-r1_R)*NBlades # 3D force in axial direction
        load3Dtan =ftan*Radius*(r2_R-r1_R)*NBlades # 3D force in azimuthal/tangential direction (not used here)
      
        # calculate thrust coefficient at the streamtube 
        CT = load3Daxial/(0.5*Area*Uinf**2)
        
        # calculate new axial induction, accounting for Glauert's correction
        anew =  ainduction(CT)
        
        # correct new axial induction with Prandtl's correction
        Prandtl, Prandtltip, Prandtlroot = PrandtlTipRootCorrection(r_R, rootradius_R, tipradius_R, Omega*Radius/Uinf, NBlades, anew)
        if (Prandtl < 0.0001): 
            Prandtl = 0.0001 # avoid divide by zero
        anew = anew/Prandtl # correct estimate of axial induction
        a = 0.75*a+0.25*anew # for improving convergence, weigh current and previous iteration of axial induction

        # calculate aximuthal induction
        aline = ftan*NBlades/(2*np.pi*Uinf*(1-a)*Omega*2*(r_R*Radius)**2)
        aline =aline/Prandtl # correct estimate of azimuthal induction with Prandtl's correction

        if iteration > 5000:
            print("Warning: iteration process did not converge after 5000 iterations, with error limit of ", Erroriterations)
            break

    return [a, aline, r_R, fnorm, ftan, gamma, alpha, phi]

# --------- Module 4 : BEM executor ---------
def visualiser(results, Uinf, Radius, tsr, output_dir=None):
    fig1 = plt.figure(figsize=(12,6))
    plt.title(f'Spanwise distribution of angles (TSR = {tsr})')
    plt.plot(results[:,2], results[:,6], 'b-', label='Angle of attack (deg)')
    plt.plot(results[:,2], results[:,7], 'r--', label='Inflow angle (deg)')
    plt.xlabel('r/R')
    plt.ylabel('Angle (deg)')
    plt.grid()
    plt.legend()
    if output_dir is not None:
        save_figure(fig1, output_dir / f"{tsr_folder_name(tsr)}_spanwise_angles.png")

    fig2 = plt.figure(figsize=(12, 6))
    plt.title(f'Axial and tangential induction (TSR = {tsr})')
    plt.plot(results[:,2], results[:,0], 'r-', label=r'$a$')
    plt.plot(results[:,2], results[:,1], 'g--', label=r'$a^,$')
    plt.grid()
    plt.xlabel('r/R')
    plt.legend()
    if output_dir is not None:
        save_figure(fig2, output_dir / f"{tsr_folder_name(tsr)}_induction.png")

    fig3 = plt.figure(figsize=(12, 6))
    plt.title(rf'Normal and tangential force, non-dimensioned by $\frac{{1}}{{2}} U_\infty^2 R$ (TSR = {tsr})')
    plt.plot(results[:,2], results[:,3]/(0.5*Uinf**2*Radius), 'r-', label=r'Fnorm')
    plt.plot(results[:,2], results[:,4]/(0.5*Uinf**2*Radius), 'g--', label=r'Ftan')
    plt.grid()
    plt.xlabel('r/R')
    plt.legend()
    if output_dir is not None:
        save_figure(fig3, output_dir / f"{tsr_folder_name(tsr)}_forces.png")


def visualiser_combined(results_by_tsr, Uinf, Radius, output_dir):
    """Create one combined figure per analysis category with all TSR curves."""
    if not results_by_tsr:
        return

    tsr_values = list(results_by_tsr.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(tsr_values)))

    fig1 = plt.figure(figsize=(12, 6))
    plt.title("Spanwise distribution of angles (all TSR cases)")
    for tsr, color in zip(tsr_values, colors):
        results = results_by_tsr[tsr]
        plt.plot(results[:,2], results[:,6], '-', color=color, label=f'Angle of attack (TSR = {tsr})')
        plt.plot(results[:,2], results[:,7], '--', color=color, label=f'Inflow angle (TSR = {tsr})')
    plt.xlabel('r/R')
    plt.ylabel('Angle (deg)')
    plt.grid()
    plt.legend()
    save_figure(fig1, output_dir / "combined_spanwise_angles.png")

    fig2 = plt.figure(figsize=(12, 6))
    plt.title('Axial and tangential induction (all TSR cases)')
    for tsr, color in zip(tsr_values, colors):
        results = results_by_tsr[tsr]
        plt.plot(results[:,2], results[:,0], '-', color=color, label=rf'$a$ (TSR = {tsr})')
        plt.plot(results[:,2], results[:,1], '--', color=color, label=rf"$a^,$ (TSR = {tsr})")
    plt.grid()
    plt.xlabel('r/R')
    plt.legend()
    save_figure(fig2, output_dir / "combined_induction.png")

    fig3 = plt.figure(figsize=(12, 6))
    plt.title(r'Normal and tangential force, non-dimensioned by $\frac{1}{2} U_\infty^2 R$ (all TSR cases)')
    for tsr, color in zip(tsr_values, colors):
        results = results_by_tsr[tsr]
        plt.plot(results[:,2], results[:,3]/(0.5*Uinf**2*Radius), '-', color=color, label=f'Fnorm (TSR = {tsr})')
        plt.plot(results[:,2], results[:,4]/(0.5*Uinf**2*Radius), '--', color=color, label=f'Ftan (TSR = {tsr})')
    plt.grid()
    plt.xlabel('r/R')
    plt.legend()
    save_figure(fig3, output_dir / "combined_forces.png")

# BLOCK 4.1 : Execute BEM solver for a given set of input parameters, and return the distribution of axial induction, tangential induction, loads and circulation along the blade
def executeBEM(Uinf, TSR, RootLocation_R, TipLocation_R , Omega, Radius, NBlades, r_R, chord_distribution, twist_distribution, polar_alpha, polar_cl, polar_cd, plot_results=True, output_dir=None):
    results = np.zeros([len(r_R)-1, 8])

    for i in range(len(r_R)-1):
        chord = np.interp((r_R[i]+r_R[i+1])/2, r_R, chord_distribution)
        twist = np.interp((r_R[i]+r_R[i+1])/2, r_R, twist_distribution)
        results[i,:] = solveStreamtube(Uinf, r_R[i], r_R[i+1], RootLocation_R, TipLocation_R , Omega, Radius, NBlades, chord, twist, polar_alpha, polar_cl, polar_cd)

    areas = (r_R[1:]**2-r_R[:-1]**2)*np.pi*Radius**2
    dr = (r_R[1:]-r_R[:-1])*Radius
    CT = np.sum(dr*results[:,3]*NBlades/(0.5*Uinf**2*np.pi*Radius**2))
    CP = np.sum(dr*results[:,4]*results[:,2]*NBlades*Radius*Omega/(0.5*Uinf**3*np.pi*Radius**2))

    print("CT is ", CT)
    print("CP is ", CP)

    if plot_results:
        visualiser_output_dir = output_dir if output_dir is not None else None
        visualiser(results, Uinf, Radius, TSR, output_dir=visualiser_output_dir)
    
    n = Omega/(2*np.pi)
    D = 2*Radius
    J = Uinf/(n*D)

    print("Advance ratio J =", J)

    rho = 1.225
    Thrust = 0.5 * rho * Uinf**2 * np.pi * Radius**2 * CT
    print("Total thrust =", Thrust, "N")

    Power = 0.5 * rho * Uinf**3 * np.pi * Radius**2 * CP
    Torque = Power / Omega

    print("Power =", Power, "W")
    print("Torque =", Torque, "Nm")
    

    return CT, CP, results, Thrust, Torque, J


# --------- Module 5 : Visualiser ---------
"""
# BLOCK 5.1 :plot Prandtl tip, root and combined correction for a number of blades and induction 'a', over the non-dimensioned radius
Prandtl, Prandtltip, Prandtlroot = PrandtlTipRootCorrection(r_R, blade_start/Radius, 1, TSR[0], blades, a)

fig1 = plt.figure(figsize=(12, 6))
plt.plot(r_R, Prandtl, 'r-', label='Prandtl')
plt.plot(r_R, Prandtltip, 'g.', label='Prandtl tip')
plt.plot(r_R, Prandtlroot, 'b.', label='Prandtl root')
plt.xlabel('r/R')
plt.legend()
"""

"""
# BLOCK 5.2 : plot airfoil polars as a function of angle of attack
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].plot(polar_alpha, polar_cl)
axs[0].set_xlim([-30,30])
axs[0].set_xlabel(r'$\alpha$')
axs[0].set_ylabel(r'$C_l$')
axs[0].grid()
axs[1].plot(polar_cd, polar_cl)
axs[1].set_xlim([0,.1])
axs[1].set_xlabel(r'$C_d$')
axs[1].grid()
plt.show()
"""

def influence_annuli(tsr):
    elements = [5, 10, 20, 50, 100, 200, 500, 1000] # number of annuli to divide blade into, for convergence analysis
    CTlist = np.zeros(len(elements))
    CPlist = np.zeros(len(elements))
    omega_convergence = tsr * U0 / Radius

    for i in range(len(elements)):
        r_R, chord_distribution, twist_distribution, a, aline = initialise(elements[i]) # initialize blade element positions and distributions for given number of blade elements
        CT, CP, results, Thrust, Torque, J = executeBEM(U0, tsr, RootLocation_R, TipLocation_R,
            omega_convergence, Radius, blades, r_R, chord_distribution, twist_distribution, polar_alpha, polar_cl, polar_cd, plot_results=False)

        CTlist[i] = CT
        CPlist[i] = CP

    return np.array(elements), CTlist, CPlist


def plot_convergence_combined(convergence_results, output_dir):
    """Create convergence figures from provided TSR cases."""
    if not convergence_results:
        return

    tsr_values = list(convergence_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(tsr_values)))

    fig_ct = plt.figure(figsize=(12, 6))
    for tsr, color in zip(tsr_values, colors):
        elements, ct_values, _ = convergence_results[tsr]
        plt.plot(elements, ct_values, 'o-', color=color, label=f'TSR = {tsr}')
    plt.xlabel('Number of annuli')
    plt.ylabel('Thrust coefficient (CT)')
    plt.title('Convergence of CT vs number of annuli (all TSR cases)')
    plt.grid()
    plt.legend()
    save_figure(fig_ct, output_dir / 'combined_convergence_CT.png')

    fig_cp = plt.figure(figsize=(12, 6))
    for tsr, color in zip(tsr_values, colors):
        elements, _, cp_values = convergence_results[tsr]
        plt.plot(elements, cp_values, 's--', color=color, label=f'TSR = {tsr}')
    plt.xlabel('Number of annuli')
    plt.ylabel('Power coefficient (CP)')
    plt.title('Convergence of CP vs number of annuli (all TSR cases)')
    plt.grid()
    plt.legend()
    save_figure(fig_cp, output_dir / 'combined_convergence_CP.png')

def main():
    prepare_output_directory(FIGURES_DIR)
    save_airfoil_polars(FIGURES_DIR)

    iter_history = np.array([]) # for storing history of iteration process, for visualisation purposes
    #elements = [5,10,20,50,100,200] # number of annuli to divide blade into, for convergence analysis
    CTlist = np.zeros(len(TSR))
    CPlist = np.zeros(len(TSR))
    Thrust_list = np.zeros(len(TSR))
    Torque_list = np.zeros(len(TSR))
    J_list = np.zeros(len(TSR))
    results_by_tsr = {}

    for j in range(len(TSR)):
        r_R, chord_distribution, twist_distribution, a, aline = initialise(100) # initialize blade element positions and distributions for 100 blade elements
        tsr_output_dir = FIGURES_DIR / tsr_folder_name(TSR[j])
        CT, CP, results, Thrust, Torque, J = executeBEM(U0, TSR[j], RootLocation_R, TipLocation_R,
        Omega[j], Radius, blades, r_R, chord_distribution, twist_distribution, polar_alpha, polar_cl, polar_cd, plot_results=False, output_dir=tsr_output_dir)
        save_prandtl_distribution(results[:,2], results[:,0], TSR[j], tsr_output_dir)
        results_by_tsr[TSR[j]] = results

        CTlist[j] = CT
        CPlist[j] = CP
        Thrust_list[j] = Thrust
        Torque_list[j] = Torque
        J_list[j] = J
        
    iter_history = np.append(iter_history, [CTlist, CPlist]) 

    if visualise:
        visualiser_combined(results_by_tsr, U0, Radius, FIGURES_DIR)

    if visualise == True:
        fig_general, ax1 = plt.subplots(figsize=(12,6))
        ax2 = ax1.twinx()

        line1, = ax1.plot(TSR/J_list, Thrust_list, 'bo-', label='Total Thrust')
        line2, = ax2.plot(TSR/J_list, Torque_list, 'rs--', label='Total Torque')

        ax1.set_xlabel('TSR/J')
        ax1.set_ylabel('Total Thrust (N)', color='b')
        ax2.set_ylabel('Total Torque (Nm)', color='r')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        ax1.set_title('Rotor loads vs TSR/J (all TSR cases)')
        ax1.grid()
        ax1.legend([line1, line2], [line1.get_label(), line2.get_label()], loc='best')

        save_figure(fig_general, FIGURES_DIR / "rotor_loads_vs_tsr_over_j.png")
        plt.show()

    convergence_results = {}
    convergence_tsr = 8
    if convergence_tsr in TSR:
        convergence_results[convergence_tsr] = influence_annuli(convergence_tsr)
    else:
        print(f"Warning: TSR={convergence_tsr} not found in TSR list; using TSR={TSR[0]} for convergence study.")
        convergence_results[TSR[0]] = influence_annuli(TSR[0])

    if visualise:
        plot_convergence_combined(convergence_results, FIGURES_DIR)

if __name__ == "__main__":
    main()