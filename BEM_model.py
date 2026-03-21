"""
Blade Element Momentum (BEM) Theory Model for a Wind Turbine Rotor
Authors: Thijmen God, Boudewijn van der Waal, Rens van Lierop
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# --------- Module 0 : Initialistation ---------


# BLOCK 0.1 : Rotor specs
Radius = 50 # m
blades = 3 # number of blades
RootLocation_R = 0.2 # m, distance from center where blades start
TipLocation_R = 1.0 # m, distance from center where blades end
blade_pitch = -2 # degrees, pitch angle at the root of the blade
visualise = False # whether to visualise results of BEM solver
convergence = False # whether to run convergence study for number of annuli


def generate_spacing(N, method='constant'):
    """Generates non-dimensional radial nodes r/R."""
    if method == 'constant':
        return np.linspace(RootLocation_R, TipLocation_R, N + 1)
    elif method == 'cosine':
        phi = np.linspace(0, np.pi, N + 1)
        nodes = 0.5 * (1 - np.cos(phi))
        # Ensure it scales between RootLocation_R and TipLocation_R
        return RootLocation_R + (TipLocation_R - RootLocation_R) * nodes

def get_blade_geometry(r_R):
    """Calculates chord and twist distributions for a given radial array."""
    chord = 3 * (1 - r_R) + 1  # (3*(1-r/R)+1) m
    twist = 14 * (1 - r_R)      # 14*(1-r/R) deg
    return chord, twist


# BLOCK 0.4 : Operational specs
U0 = 10 # m/s, free stream velocity
#TSR = [6, 8, 10] # Tip Speed Ratios to analyze
TSR = [8]
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
    Erroriterations = 0.001 # error limit for iteration process, in absolute value of induction
    a_history = [a] # history of axial induction for convergence analysis

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
        if Prandtl < 0.05: 
            Prandtl = 0.05 # avoid divide by zero
        anew = anew/Prandtl # correct estimate of axial induction
        anew = np.clip(anew, 0, 0.95) # ensure axial induction is between 0 and 1
        a = 0.85*a+0.15*anew # for improving convergence, weigh current and previous iteration of axial induction

        a_history.append(a)

        # calculate aximuthal induction
        aline = ftan*NBlades/(2*np.pi*Uinf*(1-a)*Omega*2*(r_R*Radius)**2)
        aline =aline/Prandtl # correct estimate of azimuthal induction with Prandtl's correction

        if iteration > 10000:
            print("Warning: iteration process did not converge after 10000 iterations, with error limit of ", Erroriterations)
            break

        final_data = [a, aline, r_R, fnorm, ftan, gamma, alpha, phi]

    return final_data, a_history

# --------- Module 4 : BEM executor ---------
def visualiser(results, Uinf, Radius):
    fig1 = plt.figure(figsize=(12,6))
    plt.title('Spanwise distribution of angles')
    plt.plot(results[:,2], results[:,6], 'b-', label='Angle of attack (deg)')
    plt.plot(results[:,2], results[:,7], 'r--', label='Inflow angle (deg)')
    plt.xlabel('r/R')
    plt.ylabel('Angle (deg)')
    plt.grid()
    plt.legend()

    fig2 = plt.figure(figsize=(12, 6))
    plt.title('Axial and tangential induction')
    plt.plot(results[:,2], results[:,0], 'r-', label=r'$a$')
    plt.plot(results[:,2], results[:,1], 'g--', label=r'$a^,$')
    plt.grid()
    plt.xlabel('r/R')
    plt.legend()

    fig3 = plt.figure(figsize=(12, 6))
    plt.title(r'Normal and tangential force, non-dimensioned by $\frac{1}{2} \rho U_\infty^2 R$')
    plt.plot(results[:,2], results[:,3]/(0.5*Uinf**2*Radius), 'r-', label=r'Fnorm')
    plt.plot(results[:,2], results[:,4]/(0.5*Uinf**2*Radius), 'g--', label=r'Ftan')
    plt.grid()
    plt.xlabel('r/R')
    plt.legend()

def convergence_study():
    elements = [10, 50, 200]
    plt.figure(figsize=(12,6))

    for N in elements:
        r_R = generate_spacing(N)
        chord, twist = get_blade_geometry(r_R)
        
        # Capture the new 'all_histories' output
        _, _, _, _, _, _, all_histories = executeBEM(U0, TSR[0], RootLocation_R, TipLocation_R, Omega[0], Radius, blades, r_R, chord, twist, polar_alpha, polar_cl, polar_cd)
        
        # Pick the history of an annulus near the tip (e.g., the last one)
        tip_history = all_histories[-1] 
        
        plt.plot(tip_history, label=f'{N} annuli')

    plt.xlabel('Iteration')
    plt.ylabel('Axial Induction Factor $a$')
    plt.title('Iterative Convergence History')
    plt.grid()
    plt.legend()
    plt.show()


# BLOCK 4.1 : Execute BEM solver for a given set of input parameters, and return the distribution of axial induction, tangential induction, loads and circulation along the blade
def executeBEM(Uinf, TSR, RootLocation_R, TipLocation_R , Omega, Radius, NBlades, r_R, chord_distribution, twist_distribution, polar_alpha, polar_cl, polar_cd):
    results = np.zeros([len(r_R)-1, 8])
    all_histories = [] # to store the history of axial induction for each annulus for convergence analysis

    for i in range(len(r_R)-1):
        chord = np.interp((r_R[i]+r_R[i+1])/2, r_R, chord_distribution)
        twist = np.interp((r_R[i]+r_R[i+1])/2, r_R, twist_distribution)
        final_data, a_history = solveStreamtube(Uinf, r_R[i], r_R[i+1], RootLocation_R, TipLocation_R , Omega, Radius, NBlades, chord, twist, polar_alpha, polar_cl, polar_cd)
        results[i,:] = final_data
        all_histories.append(a_history)

    areas = (r_R[1:]**2-r_R[:-1]**2)*np.pi*Radius**2
    dr = (r_R[1:]-r_R[:-1])*Radius
    CT = np.sum(dr*results[:,3]*NBlades/(0.5*Uinf**2*np.pi*Radius**2))
    CP = np.sum(dr*results[:,4]*results[:,2]*NBlades*Radius*Omega/(0.5*Uinf**3*np.pi*Radius**2))

    print("CT is ", CT)
    print("CP is ", CP)

    if visualise == True:
        visualiser(results, Uinf, Radius)
    
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
    

    return CT, CP, results, Thrust, Torque, J, all_histories


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

def influence_annuli():
    elements = [10, 20, 50, 100, 200, 500]
    methods = ['constant', 'cosine']
    results_dict = {m: {'CT': [], 'CP': []} for m in methods}
    tsr_target = 8 # Recommended target for wind turbine analysis 
    omega_target = tsr_target * U0 / Radius

    for method in methods:
        for N in elements:
            # 1. Generate specific grid
            r_R = generate_spacing(N, method)
            chord, twist = get_blade_geometry(r_R)
            
            # 2. Run BEM
            CT, CP, _, _, _, _, all_histories = executeBEM(U0, tsr_target, RootLocation_R, 
                                            TipLocation_R, omega_target, Radius, 
                                            blades, r_R, chord, twist, 
                                            polar_alpha, polar_cl, polar_cd)
            results_dict[method]['CT'].append(CT)
            results_dict[method]['CP'].append(CP)

    # Plotting logic remains similar but uses results_dict...

    if visualise:
        plt.figure(figsize=(12,6))
        plt.plot(elements, results_dict['constant']['CT'], 'ro-', label='Thrust coefficient for constant spacing')
        plt.plot(elements, results_dict['cosine']['CT'], 'bo-', label='Thrust coefficient for cosine spacing')
        plt.xlabel('Number of annuli')
        plt.ylabel('Coefficient')
        plt.title('Convergence of BEM Results')
        plt.grid()
        plt.legend()

        plt.figure(figsize=(12,6))
        plt.plot(elements, results_dict['constant']['CP'], 'ro-', label='Power coefficient for constant spacing')
        plt.plot(elements, results_dict['cosine']['CP'], 'bo-', label='Power coefficient for cosine spacing')
        plt.xlabel('Number of annuli')
        plt.ylabel('Coefficient')
        plt.title('Convergence of BEM Results')
        plt.grid()
        plt.legend()

        plt.show()




def main():
    iter_history = np.array([]) 
    CTlist = np.zeros(len(TSR))
    CPlist = np.zeros(len(TSR))
    Thrust_list = np.zeros(len(TSR))
    Torque_list = np.zeros(len(TSR))
    J_list = np.zeros(len(TSR))

    # Use a standard 100-element constant grid for the baseline results
    N_elements = 100
    r_R = generate_spacing(N_elements, method='constant')
    chord, twist = get_blade_geometry(r_R)

    for j in range(len(TSR)):
        print(f"Executing BEM for TSR = {TSR[j]}")
        CT, CP, results, Thrust, Torque, J, all_histories = executeBEM(
            U0, TSR[j], RootLocation_R, TipLocation_R,
            Omega[j], Radius, blades, r_R, chord, twist, 
            polar_alpha, polar_cl, polar_cd
        )

        CTlist[j] = CT
        CPlist[j] = CP
        Thrust_list[j] = Thrust
        Torque_list[j] = Torque
        J_list[j] = J
        
    # Run the convergence study required for the report 
    influence_annuli()
    convergence_study()


    if visualise == True:
        # Plotting logic for results...
        plt.figure(figsize=(12,6))
        plt.plot(TSR/J_list, Thrust_list, 'bo-', label='Total Thrust')
        plt.plot(TSR/J_list, Torque_list, 'rs--', label='Total Torque')
        plt.xlabel('Advance Ratio J')
        plt.ylabel('Load')
        plt.title('Rotor loads vs Advance Ratio')
        plt.grid()
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()