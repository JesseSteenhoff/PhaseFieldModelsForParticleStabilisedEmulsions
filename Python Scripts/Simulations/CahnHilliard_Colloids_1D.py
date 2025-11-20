# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 10:16:45 2025

@author: J.M. Steenhoff
"""
#Import all the required modules 
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from time import process_time
from scipy.optimize import fsolve
from scipy.integrate import simpson 
from matplotlib.legend_handler import HandlerTuple

#Preformat figures
plt.rcParams['figure.dpi'] = 300
plt.style.use(['science','no-latex'])

#%% Define simulation classes 

#The 'Liquid' class creates a phase-field object that evolves in accordance with a nondimensionalised Cahn-Hilliard equation 
class Liquid:
    
    #Binary interaction parameter
    chi=3                                 
        
    #Simulation timestep 
    dt=0.001
    
    #Stencil (grid) spacing  
    h=0.5
    
    #Initialisation method that creates a field of composition phi0, including some random thermal noise
    def __init__(self,Size,phi0,alpha):
        self.Size=Size
        self.phi0=phi0
        self.phi=self.phi0+np.random.randint(-10,10,(self.Size))*0.001
        
        #Attachment parameter
        self.alpha=alpha
        
        #Solve the binodal equation to find the equilibrium compositions 
        self.phimax=fsolve(lambda x: np.log(x/(1-x))+self.chi*(1-2*x),0.99)
        self.phimin=1-self.phimax
    
    #Method that calculates the Laplacian of an input field 'F' via central finite-difference (3-point stencil)
    def Calc_Laplacian(self,F):               
        
        #Dimensions of the input field 
        Nx=len(F)
        
        #Create templates for the Laplacian contributions  
        Laplacian_x=np.zeros(F.shape)
        
        #Apply no-flux boundary conditions along the x-direction 
        Laplacian_x[1:Nx-1]=(-2*F[1:Nx-1]+F[2:Nx]+F[0:Nx-2])/(self.h**2)
        Laplacian_x[0]=(-1*F[0]+F[1])/(self.h**2)                              
        Laplacian_x[Nx-1]=(-1*F[Nx-1]+F[Nx-2])/(self.h**2)
                   
        Laplacian=Laplacian_x
        
        return Laplacian
    
    #Method that calculates the gradient vector for an input field 'F' via central finite-difference (3-point stencil)
    def Calc_Gradient(self,F):            
        
        #Dimensions of the input field 
        Nx=len(F)
        
        #Create templates for the gradient vector components  
        Gradientx=np.zeros(F.shape)                                                     
        
        #Apply no-flux boundary conditions along the x-direction 
        Gradientx[1:Nx-1]=(F[2:Nx]-F[0:Nx-2])/(2*self.h)
        Gradientx[0]=(F[1]-F[0])/(2*self.h)                                                        
        Gradientx[Nx-1]=(F[Nx-1]-F[Nx-2])/(2*self.h)
         
        Gradient=Gradientx
        
        return Gradient
    
    #Method that evolves the order parameter field according to the nondimensionalised Cahn-Hilliard equation 
    def Propagate(self,phi,psi):
        
        #Calculates the functional derivative of the free energy (chemical potential) for the liquid
        Mu=np.log(phi/(1-phi))+self.chi*(1-2*phi)-(1-self.alpha*psi)*self.Calc_Laplacian(phi)+self.alpha*(self.Calc_Gradient(phi)[0]*self.Calc_Gradient(psi)[0])
        
        #Update the order parameter field (Euler Forward)
        self.phi+=(self.Calc_Laplacian(Mu))*self.dt
    
        return self.phi

#The Colloid class describes the behaviour of surface-active nanoparticles in the system
class Colloid(Liquid):
    
    #Relative mobility of the colloids
    Mc=0.01
    
    #Initisalisation method for the Colloid field (child class of 'Liquid')
    def __init__(self,Size,psi0,alpha):
        self.Size=Size
        self.psi0=psi0
        self.psi=self.psi0+np.zeros(self.Size)
        
        #Initialise parent class
        Liquid.__init__(self,Size,phi0,alpha)
    
    #Method that propagates the colloid order parameter field in time according to the nondimensionalised Cahn-Hilliard equation 
    def Propagate(self,psi,phi):
        
        #Calculate the gradient of the liquid field 
        Gradx=self.Calc_Gradient(phi)
        
        #Calculates the functional derivative of the free energy (chemical potential) for the colloids (ideal gas approximation for bulk contributions)
        Mu=np.log(psi)-self.alpha/2*(np.abs(Gradx)**2)
        
        #Update order parameter (Euler Forward)
        self.psi+=(self.Mc*self.Calc_Laplacian(Mu))*self.dt
    
        return self.psi
    

#%% Run phase-field simulations

#Simulation dimensions 
Dimensions=20

#Initial compositions
phi0=0.50
psi0=0.50

#Attachment parameters 
alpha_range=[0,1.2,1.5]

#Lists with liquid and colloid profiles 
Liquid_list=[]
Colloid_list=[]

#Storage lists for the effective gradient energy parameter and interfacial tension
k_storage=[]
sigma_storage=[]

#Storage list for interface excess (colloids and liquid)
DeltaPhi_storage=[]
DeltaPsi_storage=[]

for alpha in alpha_range:

    #Initialise the liquid order parameter field 
    Oil=Liquid(Dimensions,phi0,alpha)
    
    #Split the system in two regions of equilibrium composition
    Oil.phi[0:int(Dimensions/2)]=Oil.phimax
    Oil.phi[int(Dimensions/2):]=Oil.phimin
    
    #Initialise the colloid field 
    Colloids=Colloid(Dimensions,psi0,alpha)         
    
    #Total number of simulation steps 
    N=500000                     
    
    #List with the effective gradient energy parameter
    k_eff_list=[]
    
    #List with the effective interfacial tension 
    sigma_eff_list=[]
    
    #List with interface excess due to colloids
    DeltaPhi_list=[]
    
    #Start-point for determining computation time 
    t0=process_time()  
    
    for n in range(N+1):
        
        #Create local copies of the liquid and colloid field 
        phi,psi=np.copy(Oil.phi),np.copy(Colloids.psi)
        
        #Calculate the effective gradient energy parameter 
        k_eff=1-Oil.alpha*Colloids.psi
        k_eff_list.append(k_eff)
        
        #Calculate the effective interfacial tension 
        sigma_eff=simpson(k_eff*(Oil.Calc_Gradient(phi))**2,dx=Oil.h)
        sigma_eff_list.append(sigma_eff)
        
        #Calculate the interface excess due to colloid interaction 
        DeltaPhi=Oil.phi[9]-Oil.phimax
        DeltaPhi_list.append(DeltaPhi)
        
        #Propagate the fields in time via the Cahn-Hilliard equation
        Oil.Propagate(phi,psi)
        Colloids.Propagate(psi,phi)
        
        #Check progress 
        if n%(N/10)==0:                
            print(str(100*n/N)+'% complete')
            
    
    #End-point for determining computation time
    t1=process_time()   
    #Total computation time
    Deltat=t1-t0         
    print('Computation time: '+str(Deltat)+' s')
    
    #Append the calculated profiles of both liquid and colloid fields 
    Liquid_list.append(Oil.phi)
    Colloid_list.append(Colloids.psi)
    
    #Append the list of the effective gradient energy parameter and interfacial tension 
    k_storage.append(k_eff_list)
    sigma_storage.append(sigma_eff_list)
    
    #Append the list of the interface excess
    DeltaPhi_storage.append(DeltaPhi_list)
    
    #Calculate and append the relative achieved colloid accumulation on the interface
    DeltaPsi=Colloids.psi[9]-Colloids.psi[0]
    DeltaPsi_storage.append(DeltaPsi)

#%%Plot liquid and colloid profiles for different values of the attachment parameter 

#Select desired colours from standard colour cycle
selection=[0,1,3]
if len(selection)>len(alpha_range):
    print('Not enough colours!')
colors=[]
for c in selection:
    colors.append(plt.rcParams['axes.prop_cycle'].by_key()['color'][c])

#x-spacing (reduced units)
x=np.arange(0,Dimensions,Oil.h)

#Plot liquid and colloid profiles in separate subfigures
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
plt.xlabel(r'Position, $\tilde{x}$')
axes[0].set_ylabel('Liquid, $\phi$')
axes[1].set_ylabel('Nanoparticle, $\psi$')

plt.minorticks_on()
plt.subplots_adjust(hspace=0.03)

#Storage for label handles 
Handles=[]

for i in range(len(alpha_range)):
    p1,=axes[0].plot(x[5:15],Liquid_list[i][5:15],marker='^',linestyle='--',markersize=3,color=colors[i])
    p2,=axes[1].plot(x[5:15],Colloid_list[i][5:15],marker='o',linestyle='--',markersize=3,color=colors[i])

    Handles.append((p1,p2))

#Plot vertical lines to indicate center of interface
axes[0].vlines(4.75,-0.05,1.05,linestyle='--',color='black',alpha=0.25)
axes[1].vlines(4.75,0.4,1,linestyle='--',color='black',alpha=0.25)

axes[0].set_ylim(-0.05,1.05)
axes[1].set_ylim(0.40,0.90)

#Construct the legend
fig.legend(Handles,alpha_range,handler_map={tuple: HandlerTuple(ndivide=None)},fontsize=9,loc=(0.755,0.635),title=r'$\tilde{\alpha}$')

#%% For selected attachment parameters, visualise the evolution of the interfacial tension and the interfacial excess over time 

#Find corresponding profiles of the interfacial tension and interfacial excess 
sigmas=np.array(sigma_storage)
DeltaPhis=np.array(DeltaPhi_storage)

#Calculate the simulated time (nondimensional)
t_range=np.arange(0,Oil.dt*N+Oil.dt,Oil.dt)

#Construct Figure frame 
fig, ax1 = plt.subplots()

ax1.set_xlabel(r'Simulated time, $\tilde{t}$')
ax2 = ax1.twinx()
ax1.set_ylabel(r'Interfacial tension, $\tilde{\sigma}$')
ax2.set_ylabel(r'Interfacial excess, $\Delta \phi$')

ax1.set_ylim(-0.4, 0.4)
ax2.set_ylim(-0.4, 0.4)
plt.xlim(0,75)

#Plot horizontal line where y=0 for both axes
plt.hlines(0,0,75,linestyle='-',color='black',alpha=0.75)

#Plot interfacial tension and interfacial excess 
for i in range(len(alpha_range)):
    
    ax1.plot(t_range,sigmas[i],color=colors[i],label=str(np.round(alpha_range[i],2)))
    ax2.plot(t_range,DeltaPhis[i],linestyle='--',color=colors[i])

#Construct legend
ax1.legend(title=r'$\tilde{\alpha}$',ncol=3,loc=(0.1,1))

plt.text(17,0.065,r'$\tilde{\sigma}$',color=colors[2])
plt.text(15,-0.085,r'$\Delta\phi$',color=colors[2])



