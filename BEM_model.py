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
blade_start = 0.2 * Radius # m, distance from center where blades start
blade_pitch = -2 # degrees, pitch angle at the root of the blade

# BLOCK 0.2 : Section streamtubes
delta_r_R = 0.01 # non-dimensioned width of blade element, in fraction of rotor radius
r_R = np.arange(blade_start/Radius, 1 + delta_r_R, delta_r_R) # non-dimensioned radial position of blade element, in fraction of rotor radius
chord_distribution = 3*(1-r_R) + 1 # m, chord distribution along the blade, linearly decreasing from 4 m at the root to 1 m at the tip
twist_distribution = 14*(1-r_R) # degrees, twist distribution along the blade,

# BLOCK 0.4 : Operational specs
U0 = 10 # m/s, free stream velocity
TSR = [6, 8, 10] # Tip Speed Ratios to analyze
rotor_yaw = 0 # degrees, yaw angle of the rotor
Omega = [TSR[i]*U0/Radius for i in range(len(TSR))] # rotational velocity of the rotor, calculated from TSR and free stream velocity
glauert_correction = True # whether to apply Glauert's correction for heavily loaded rotors
prandtl_correction = True # whether to apply Prandtl's tip loss correction

# BLOCK 0.5 : Prescribe first estimate induction factor for all streamtubes
a = np.zeros(np.shape(r_R))+0.3 # axial induction factor, initial estimate
aline = np.zeros(np.shape(r_R)) # tangential induction factor, initial estimate


# --------- Module 1 : Lift and Drag Coefficients ---------

# BLOCK 1.1 : Import airfoil data and polar
airfoil = 'polar DU95W180.xlsx'
data=pd.read_excel(airfoil, header=0,
                    names = ["alpha", "cl", "cd", "cm"]).dropna()
data = data.drop(data.index[0]) # drop first row with header info
polar_alpha = data['alpha'][:]
polar_cl = data['cl'][:]
polar_cd = data['cd'][:]


# --------- Module 2 : Normal and Tangential Forces ---------

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
    alpha = twist + inflowangle*180/np.pi
    cl = np.interp(alpha, polar_alpha, polar_cl)
    cd = np.interp(alpha, polar_alpha, polar_cd)
    lift = 0.5*vmag2*cl*chord
    drag = 0.5*vmag2*cd*chord
    fnorm = lift*np.cos(inflowangle)+drag*np.sin(inflowangle)
    ftan = lift*np.sin(inflowangle)-drag*np.cos(inflowangle)
    gamma = 0.5*np.sqrt(vmag2)*cl*chord
    return fnorm , ftan, gamma

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
    
    Niterations = 100
    Erroriterations =0.00001 # error limit for iteration process, in absolute value of induction
    
    for i in range(Niterations):
        # calculate velocity and loads at blade element
        Urotor = Uinf*(1-a) # axial velocity at rotor
        Utan = (1+aline)*Omega*r_R*Radius # tangential velocity at rotor
        # calculate loads in blade segment in 2D (N/m)
        fnorm, ftan, gamma = loadBladeElement(Urotor, Utan, r_R,chord, twist, polar_alpha, polar_cl, polar_cd)
        load3Daxial =fnorm*Radius*(r2_R-r1_R)*NBlades # 3D force in axial direction
        load3Dtan =ftan*Radius*(r2_R-r1_R)*NBlades # 3D force in azimuthal/tangential direction (not used here)
      
        # calculate thrust coefficient at the streamtube 
        CT = load3Daxial/(0.5*Area*Uinf**2)
        
        # calculate new axial induction, accounting for Glauert's correction
        anew =  ainduction(CT)
        
        # correct new axial induction with Prandtl's correction
        Prandtl, Prandtltip, Prandtlroot = PrandtlTipRootCorrection(r_R, rootradius_R, tipradius_R, Omega*Radius/Uinf, NBlades, anew);
        if (Prandtl < 0.0001): 
            Prandtl = 0.0001 # avoid divide by zero
        anew = anew/Prandtl # correct estimate of axial induction
        a = 0.75*a+0.25*anew # for improving convergence, weigh current and previous iteration of axial induction

        # calculate aximuthal induction
        aline = ftan*NBlades/(2*np.pi*Uinf*(1-a)*Omega*2*(r_R*Radius)**2)
        aline =aline/Prandtl # correct estimate of azimuthal induction with Prandtl's correction
        
        # test convergence of solution, by checking convergence of axial induction
        if (np.abs(a-anew) < Erroriterations): 
            break

    return [a , aline, r_R, fnorm , ftan, gamma]

# --------- Module 4 : Calculate Loads and Power ---------





# --------- Module 4 : Visualiser ---------
"""
# BLOCK 4.1 :plot Prandtl tip, root and combined correction for a number of blades and induction 'a', over the non-dimensioned radius
Prandtl, Prandtltip, Prandtlroot = PrandtlTipRootCorrection(r_R, blade_start/Radius, 1, TSR[0], blades, a)

fig1 = plt.figure(figsize=(12, 6))
plt.plot(r_R, Prandtl, 'r-', label='Prandtl')
plt.plot(r_R, Prandtltip, 'g.', label='Prandtl tip')
plt.plot(r_R, Prandtlroot, 'b.', label='Prandtl root')
plt.xlabel('r/R')
plt.legend()
"""

# BLOCK 4.2 : plot airfoil polars as a function of angle of attack
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

# --------- Module 5 : BEM Executor ---------

# BLOCK 5.1 : Execute BEM solver for a given set of input parameters, and return the distribution of axial induction, tangential induction, loads and circulation along the blade
def executeBEM(Uinf, TSR, rootradius_R, tipradius_R , Omega, Radius, NBlades, chord_distribution, twist_distribution, polar_alpha, polar_cl, polar_cd):
    a_distribution = np.zeros(np.shape(r_R))
    aline_distribution = np.zeros(np.shape(r_R))
    fnorm_distribution = np.zeros(np.shape(r_R))
    ftan_distribution = np.zeros(np.shape(r_R))
    gamma_distribution = np.zeros(np.shape(r_R))

    for i in range(len(r_R)):
        r1_R = r_R[i]-delta_r_R/2
        r2_R = r_R[i]+delta_r_R/2
        chord = chord_distribution[i]
        twist = twist_distribution[i]
        a_distribution[i], aline_distribution[i], r_R[i], fnorm_distribution[i], ftan_distribution[i], gamma_distribution[i] = solveStreamtube(Uinf, r1_R, r2_R, rootradius_R, tipradius_R , Omega, Radius, NBlades, chord, twist, polar_alpha, polar_cl, polar_cd)

    return a_distribution, aline_distribution, fnorm_distribution, ftan_distribution, gamma_distribution

results = np.zeros([len(r_R)-1, 6, len(TSR)])

for i in range(len(TSR)):
    chord = np.interp((r_R[i]+r_R[i+1])/2, r_R, chord_distribution)
    twist = np.interp((r_R[i]+r_R[i+1])/2, r_R, twist_distribution)

    results[:,:,i] = executeBEM(U0, TSR[i], blade_start/Radius, 1 , Omega[i], Radius, blades, chord_distribution, twist_distribution, polar_alpha, polar_cl, polar_cd)

areas = (r_R[1:]**2-r_R[:-1]**2)*np.pi*Radius**2
dr = (r_R[1:]-r_R[:-1])*Radius
CT = np.sum(dr*results[:,3]*blades/(0.5*U0**2*np.pi*Radius**2))
CP = np.sum(dr*results[:,4]*results[:,2]*blades*Radius*Omega/(0.5*U0**3*np.pi*Radius**2))

print("CT is ", CT)
print("CP is ", CP)