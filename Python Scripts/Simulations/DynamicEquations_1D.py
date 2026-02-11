# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 20:34:51 2026

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
            M = 1*np.ones(len(psi))

        return M

    # Method that evolves the order parameter field according to the dynamic equations 
    def Propagate(self, phi, psi, psic, n):
        
        #Lattice dimensions 
        Nx=len(phi)
        
        # Calculate the mobility of the liquid at lattice points
        M = self.Calc_Mobility(psi, psic, n)
        
        # Calculate mobility at lattice faces
        M_faces=np.zeros(Nx+1)
        #Harmonic mean 
        M_faces[1:Nx]=2/(1/M[1:Nx]+1/M[0:Nx-1])
      
        # Calculates the functional derivative of the free energy (chemical potential, lattice points) for the liquid
        Mu = np.log(phi/(1-phi))+self.chi*(1-2*phi)-self.Calc_Laplacian(phi)
        
        #Calculate fluxes at lattice faces
        J=np.zeros(Nx+1) #No-Flux boundary conditions!
        J[1:Nx]=-M_faces[1:Nx]*(Mu[1:Nx]-Mu[0:Nx-1])/self.h
        
        #Approximate divergence of fluxes at lattice points (finite-difference)
        DivJ=(J[1:Nx+1]-J[0:Nx])/self.h
        
        # Update order parameter field (Euler Forward)
        self.phi += -DivJ*self.dt

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
            MC = self.Mc*np.ones(len(psi))

        return MC
 
    # Method that propagates the colloid order parameter field in time according to the dynamic equations (Finite-Volume method)
    def Propagate(self, psi, phi, psic, n):
        
        #Lattice dimensions 
        Nx=len(psi)
        
        # Calculate the mobility of the colloids at lattice points
        MC = self.Calc_Mobility_C(psi, psic, n)
        
        # Calculate mobility at lattice faces
        MC_faces=np.zeros(Nx+1)
        #Harmonic mean 
        MC_faces[1:Nx]=2/(1/MC[1:Nx]+1/MC[0:Nx-1])
        
        # Calculate the gradient of the liquid field
        Gradx = self.Calc_Gradient(phi)
        
        # Calculates the functional derivative of the free energy (chemical potential, lattice points) for the colloids (ideal gas approximation for bulk contributions)
        Mu = np.log(psi)-self.alpha/2*(Gradx**2)
        
        #Calculate fluxes at lattice faces
        J=np.zeros(Nx+1) #No-Flux boundary conditions!
        J[1:Nx]=-MC_faces[1:Nx]*(Mu[1:Nx]-Mu[0:Nx-1])/self.h
        
        #Approximate divergence of fluxes at lattice points (finite-difference)
        DivJ=(J[1:Nx+1]-J[0:Nx])/self.h
        
        # Update order parameter field (Euler Forward)
        self.psi += -DivJ*self.dt
        
        return self.psi


# %% Run phase-field simulation

# Simulation dimensions
Dimensions = 20

# Initial composition
phi0 = 0.50
psi0 = 0.50

# Threshold value for colloid jamming
psi_c = 0.95
# ''Sharpness'' of colloid mobility function
s = 75

# Attachment parameters
alpha_range = [0, 5, 20]

# Lists with liquid and colloid profiles (Non-jammed)
Liquid_list = []
Colloid_list = []

# Lists with liquid and colloid profiles (jammed)
Liquid_list_jammed = []
Colloid_list_jammed = []

# List with average colloid content (jammed)
psi_av_list=[]


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
    
    #Local list with average colloid content 
    psi_av=[]

    # Simulated time
    t_sim = 2000
    # Total number of simulation steps
    N = int(t_sim/Oil.dt)

    # Start-point for determining computation time
    t0 = process_time()

    for n in range(N):

        # Create local copies of the liquid and colloid field
        phi, psi = np.copy(Oil.phi), np.copy(Colloids.psi)
        phi_jammed, psi_jammed = np.copy(
            Oil_jammed.phi), np.copy(Colloids_jammed.psi)

        # Propagate the fields in time via the Cahn-Hilliard equation
        Oil.Propagate(phi, psi, psi_c, s)
        Colloids.Propagate(psi, phi, psi_c, s)

        Oil_jammed.Propagate(phi_jammed, psi_jammed, psi_c, s)
        Colloids_jammed.Propagate(psi_jammed, phi_jammed, psi_c, s)
        
        #Calculate and append average colloid content
        psi_av.append(np.average(np.copy(Colloids_jammed.psi)))

        # Check system progress at 10% intervals
        if ((n+1)*Oil.dt)%(t_sim/10)==0:
            print(str(100*(n+1)/N)+'% complete')

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
    
    #Append average colloid content
    psi_av_list.append(psi_av)


#%% Visualise evolution of colloid content over time 

t_range=np.arange(Oil.dt,t_sim+Oil.dt,Oil.dt)

plt.figure()
plt.minorticks_on()
plt.ylabel(r'Av. Nanoparticle Content, $\psi_{av}$')
plt.xlabel(r'Simulated time, $\tilde{t}$')

for i in range(len(psi_av_list)):
    plt.plot(t_range,psi_av_list[i],label=str(alpha_range[i]))

plt.legend(title=r'$\alpha$')

plt.ylim(0.499,0.501)

# %%Plot liquid and colloid profiles for different alpha-values

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
plt.subplots_adjust(hspace=0.04)

# Storage for label handles
Handles = []

#Mask for plotting 
B=2
E=Dimensions-B

for i in range(len(alpha_range)):
    p1, = axes[0].plot(x[B:E], Liquid_list[i][B:E], marker='^',
                       linestyle='--', markersize=3, color=colors[i])
    p2, = axes[1].plot(x[B:E], Colloid_list[i][B:E], marker='o',
                       linestyle='--', markersize=3, color=colors[i])
    p3, = axes[2].plot(x[B:E], Colloid_list_jammed[i][B:E],
                       marker='o', linestyle='--', markersize=3, color=colors[i])

    Handles.append((p1, p2))

# Plot vertical lines to indicate center of interface
axes[0].vlines(4.75, -0.05, 1.05, linestyle='--', color='black', alpha=0.25)
axes[1].vlines(4.75, 0, 2.1, linestyle='--', color='black', alpha=0.25)
axes[2].vlines(4.75, 0, 2.1, linestyle='--', color='black', alpha=0.25)

axes[0].set_ylim(-0.05, 1.05)
axes[1].set_ylim(0, 2.15)
axes[2].set_ylim(0, 2.15)

# Construct the legend
fig.legend(Handles, alpha_range, handler_map={tuple: HandlerTuple(
    ndivide=None)}, fontsize=9, loc=(0.750, 0.755), title=r'$\tilde{\alpha}$')

# Put in some text
plt.text(2.3, 4.0, 'Non-jamming',ha='center')
plt.text(2.3, 1.8, 'Jamming',ha='center')

