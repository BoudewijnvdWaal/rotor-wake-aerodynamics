"""
Blade Element Momentum (BEM) Theory Model for a Wind Turbine Rotor
Authors: Thijmen God, Boudewijn van der Waal, Rens van Lierop
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# --------- Module 0 : Initialistation ---------

# Rotor specs
radius = 50 # m
blades = 3 # number of blades
blade_start = 0.2 * radius # m, distance from center where blades start
blade_pitch = -2 # degrees, pitch angle at the root of the blade
glauert_correction = True # whether to apply Glauert's correction for heavily loaded rotors
prandtl_correction = True # whether to apply Prandtl's tip loss correction

# Import airfoil data and polar
airfoil = 'polar DU95W180.xlsx'
data=pd.read_excel(airfoil, header=0,
                    names = ["alpha", "cl", "cd", "cm"]).dropna()

polar_data = data.drop(data.index[0]) # drop first row with header info

polar_alpha = data['alpha'][:]
polar_cl = data['cl'][:]
polar_cd = data['cd'][:]

# Operational specs
U0 = 10 # m/s, free stream velocity
TSR = [6, 8, 10] # Tip Speed Ratios to analyze
rotor_yaw = 0 # degrees, yaw angle of the rotor

def twist(r):
    """Calculate the twist angle at a given radius."""
    return 14 * (1 - r / radius) # degrees

def chord(r):
    """Calculate the chord length at a given radius."""
    return 3 * (1 - r / radius) + 1 # m





# --------- Module 1 : Baseline Functions ---------
def CTfunction(a):
    """
    This function calculates the thrust coefficient as a function of induction factor 'a'
    'glauert' defines if the Glauert correction for heavily loaded rotors should be used; default value is false
    """
    CT = np.zeros(np.shape(a))
    CT = 4*a*(1-a)  

    CT1=1.816
    a1=1-np.sqrt(CT1)/2
    CT[a>a1] = CT1-4*(np.sqrt(CT1)-1)*(1-a[a>a1])

    return CT

def ainduction(CT):
    """
    This function calculates the induction factor 'a' as a function of thrust coefficient CT 
    including Glauert's correction
    """
    a = np.zeros(np.shape(CT))
    CT1=1.816;
    CT2=2*np.sqrt(CT1)-CT1
    a[CT>=CT2] = 1 + (CT[CT>=CT2]-CT1)/(4*(np.sqrt(CT1)-1))
    a[CT<CT2] = 0.5-0.5*np.sqrt(1-CT[CT<CT2])
    return a



# --------- Module 2 : Corrections ---------
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





# --------- Module 3 : Visualiser ---------
# plot Prandtl tip, root and combined correction for a number of blades and induction 'a', over the non-dimensioned radius
r_R = np.arange(0.1, 1, .01)
a = np.zeros(np.shape(r_R))+0.3
Prandtl, Prandtltip, Prandtlroot = PrandtlTipRootCorrection(r_R, 0.1, 1, 7, 3, a)

fig1 = plt.figure(figsize=(12, 6))
plt.plot(r_R, Prandtl, 'r-', label='Prandtl')
plt.plot(r_R, Prandtltip, 'g.', label='Prandtl tip')
plt.plot(r_R, Prandtlroot, 'b.', label='Prandtl root')
plt.xlabel('r/R')
plt.legend()


# plot polars of the airfoil C-alfa and Cl-Cd

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



# --------- Module 99 : Actuator Disk Model ---------
"""
implement code for actuator disk theory here to answer question 2 of the assignment
"""