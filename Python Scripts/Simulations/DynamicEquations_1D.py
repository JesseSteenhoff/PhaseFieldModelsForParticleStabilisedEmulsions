# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 11:53:40 2025

@author: J.M. Steenhoff
"""
# Import all the required modules
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from time import process_time
from scipy.optimize import fsolve
from matplotlib.legend_handler import HandlerTuple

# Preformat figures
plt.rcParams['figure.dpi'] = 300
plt.style.use(['science', 'no-latex'])


# %% Define simulation classes

#The 'Liquid' class creates a phase-field object that evolves in accordance with the presented dynamic equations for jamming and non-jamming colloids
class Liquid:
    
    # Binary interaction parameter
    chi = 3
    
    # Simulation timestep
    dt = 0.002
    # Stencil (grid) spacing
    h = 0.50

    # Initialisation method that creates a field of composition phi0, including some random thermal noise
    def __init__(self, Size, phi0, alpha, jamming):
        self.Size = Size
        self.phi0 = phi0
        self.phi = self.phi0+np.random.randint(-10, 10, (self.Size))*0.001

        # Solve the binodal equation to find the equilibrium compositions
        self.phimax = fsolve(lambda x: np.log(x/(1-x))+self.chi*(1-2*x), 0.99)
        self.phimin = 1-self.phimax

        # Attachment parameter
        self.alpha = alpha

        # Set jamming of colloids (jamming and non-jamming colloids for 'True' and 'False' respectively)
        self.jamming = jamming

    # Method that calculates the Laplacian of an input field 'F' via central finite-difference (3-point stencil)
    def Calc_Laplacian(self, F):

        # Dimensions of the input field
        Nx = len(F)

        # Create templates for the Laplacian contributions
        Laplacian_x = np.zeros(F.shape)

        # Apply no-flux boundary conditions along the x-direction
        Laplacian_x[1:Nx-1] = (-2*F[1:Nx-1]+F[2:Nx]+F[0:Nx-2])/(self.h**2)
        Laplacian_x[0] = (-1*F[0]+F[1])/(self.h**2)
        Laplacian_x[Nx-1] = (-1*F[Nx-1]+F[Nx-2])/(self.h**2)

        Laplacian = Laplacian_x

        return Laplacian

    # Method that calculates the gradient vector for an input field 'F' via central finite-difference (3-point stencil)
    def Calc_Gradient(self, F):

        # Dimensions of the input field
        Nx = len(F)

        # Create templates for the gradient vector components
        Gradientx = np.zeros(F.shape)

        # Apply no-flux boundary conditions along the x-direction
        Gradientx[1:Nx-1] = (F[2:Nx]-F[0:Nx-2])/(2*self.h)
        Gradientx[0] = (F[1]-F[0])/(2*self.h)
        Gradientx[Nx-1] = (F[Nx-1]-F[Nx-2])/(2*self.h)

        Gradient = Gradientx

        return Gradient

    # Method that modulates the liquid mobility based on the presence of colloids (jamming)
    def Calc_Mobility(self, psi, psic, n):

        if self.jamming == True:
            M = 1/2*(1-np.tanh((psi-psic)*n))
        else:
            M = 1

        return M

    # Method that evolves the order parameter field according to the dynamic equations 
    def Propagate(self, phi, psi, psic, n):

        # Calculates the functional derivative of the free energy (chemical potential) for the liquid
        Mu = np.log(phi/(1-phi))+self.chi*(1-2*phi)-self.Calc_Laplacian(phi)

        # Update the order parameter field (Euler Forward)
        self.phi += (self.Calc_Mobility(psi, psic, n)*self.Calc_Laplacian(Mu))*self.dt

        return self.phi

#The Colloid class describes the behaviour of surface-active nanoparticles in the system. Colloids can be either jamming or non-jamming.
class Colloid(Liquid):

    # Relative mobility of the colloids
    Mc = 0.01

    # Initisalisation method for the Colloid field (child class of 'Liquid')
    def __init__(self, Size, psi0, alpha, jamming):

        # Initialise parent class (Liquid)
        Liquid.__init__(self, Size, phi0, alpha, jamming)

        self.psi0 = psi0
        self.psi = self.psi0+np.zeros(self.Size)

    # Method that modulates the colloid mobility (jamming)
    def Calc_Mobility_C(self, psi, psic, n):
        if self.jamming == True:
            MC = self.Mc*1/2*(1-np.tanh((psi-psic)*n))
        else:
            MC = self.Mc

        return MC

    # Method that propagates the colloid order parameter field in time according to the dynamic equations
    def Propagate(self, psi, phi, psic, n):

        # Calculate the gradient of the liquid field
        Gradx = self.Calc_Gradient(phi)

        # Calculate the mobility of the colloids
        MC = self.Calc_Mobility_C(psi, psic, n)

        # Calculates the functional derivative of the free energy (chemical potential) for the colloids (ideal gas approximation for bulk contributions)
        Mu = np.log(psi)-self.alpha/2*(Gradx**2)

        # Update order parameter field (Euler Forward)
        self.psi += (MC*self.Calc_Laplacian(Mu))*self.dt

        return self.psi


# %% Run phase-field simulation

# Simulation dimensions
Dimensions = 20

# Initial composition (liquids, colloids)
phi0 = 0.50
psi0 = 0.50

# Threshold value for colloid jamming
psi_c = 0.95
# ''Sharpness'' of colloid mobility function
s = 50

# Attachment parameters
alpha_range = [0, 5, 20]

# Lists with liquid and colloid profiles (non-jamming)
Liquid_list = []
Colloid_list = []

# Lists with liquid and colloid profiles (jamming)
Liquid_list_jammed = []
Colloid_list_jammed = []


for alpha in alpha_range:

    # Initialise the liquid order parameter field
    Oil = Liquid(Dimensions, phi0, alpha, jamming=False)
    Oil_jammed = Liquid(Dimensions, phi0, alpha, jamming=True)

    # Split the system in two regions of equilibrium composition
    Oil.phi[0:int(Dimensions/2)] = Oil.phimax
    Oil.phi[int(Dimensions/2):] = Oil.phimin

    Oil_jammed.phi[0:int(Dimensions/2)] = Oil.phimax
    Oil_jammed.phi[int(Dimensions/2):] = Oil.phimin

    # Initialise the colloid field
    Colloids = Colloid(Dimensions, psi0, alpha, jamming=False)
    Colloids_jammed = Colloid(Dimensions, psi0, alpha, jamming=True)

    # Simulated time
    t_sim = 750
    
    # Total number of simulation steps
    N = int(t_sim/Oil.dt)

    # Start-point for determining computation time
    t0 = process_time()

    for n in range(N+1):

        # Create local copies of the liquid and colloid field
        phi, psi = np.copy(Oil.phi), np.copy(Colloids.psi)
        phi_jammed, psi_jammed = np.copy(Oil_jammed.phi), np.copy(Colloids_jammed.psi)

        # Propagate the fields in time via the dynamic equations
        Oil.Propagate(phi, psi, psi_c, s)
        Colloids.Propagate(psi, phi, psi_c, s)

        Oil_jammed.Propagate(phi_jammed, psi_jammed, psi_c, s)
        Colloids_jammed.Propagate(psi_jammed, phi_jammed, psi_c, s)

        # Check progress at 10% intervals
        if n % (N/10) == 0:
            print(str(100*n/N)+'% complete')

    # End-point for determining computation time
    t1 = process_time()
    # Total computation time
    Deltat = t1-t0
    print('Computation time: '+str(Deltat)+' s')

    # Append the calculated profiles of both liquid and colloid fields
    Liquid_list.append(Oil.phi)
    Colloid_list.append(Colloids.psi)

    Liquid_list_jammed.append(Oil_jammed.phi)
    Colloid_list_jammed.append(Colloids_jammed.psi)


#%%Plot liquid and colloid profiles for different values of the attachment parameter

# Select desired colours from standard colour cycle
selection = [0, 1, 3]
if len(selection) > len(alpha_range):
    print('Not enough colours!')
colors = []
for c in selection:
    colors.append(plt.rcParams['axes.prop_cycle'].by_key()['color'][c])

# x-spacing (reduced units)
x = np.arange(0, Dimensions, Oil.h)

# Create Figure frame [cm]
W = 8.5
H = 12
# Convert to inches
W, H = np.array([W, H])*0.393700787

# Plot liquid and colloid profiles in separate subfigures
fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(W, H))
plt.xlabel(r'Position, $\tilde{x}$')
axes[0].set_ylabel('Liquid, $\phi$')
axes[1].set_ylabel('Nanoparticle, $\psi$')
axes[2].set_ylabel('Nanoparticle, $\psi$')

plt.minorticks_on()
plt.subplots_adjust(hspace=0.03)

# Storage for label handles
Handles = []

for i in range(len(alpha_range)):
    p1, = axes[0].plot(x[5:15], Liquid_list[i][5:15], marker='^',
                       linestyle='--', markersize=3, color=colors[i])
    p2, = axes[1].plot(x[5:15], Colloid_list[i][5:15], marker='o',
                       linestyle='--', markersize=3, color=colors[i])
    p3, = axes[2].plot(x[5:15], Colloid_list_jammed[i][5:15],
                       marker='o', linestyle='--', markersize=3, color=colors[i])

    Handles.append((p1, p2))

# Plot vertical lines to indicate center of interface
axes[0].vlines(4.75, -0.05, 1.05, linestyle='--', color='black', alpha=0.25)
axes[1].vlines(4.75, 0, 2.1, linestyle='--', color='black', alpha=0.25)
axes[2].vlines(4.75, 0, 1.1, linestyle='--', color='black', alpha=0.25)

axes[0].set_ylim(-0.05, 1.05)
axes[1].set_ylim(0, 2.1)
axes[2].set_ylim(0, 1.1)

# Construct the legend
fig.legend(Handles, alpha_range, handler_map={tuple: HandlerTuple(
    ndivide=None)}, fontsize=9, loc=(0.750, 0.755), title=r'$\tilde{\alpha}$')

# Put in some text
plt.text(2.45, 2.05, 'Non-jamming')
plt.text(2.45, 0.925, 'Jamming')


