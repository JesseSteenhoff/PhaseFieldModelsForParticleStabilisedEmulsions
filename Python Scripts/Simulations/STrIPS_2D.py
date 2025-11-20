# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 19:35:03 2025

@author: J.M. Steenhoff
"""
#Import all the required modules 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import scienceplots

from numba import njit,prange
from time import perf_counter
from matplotlib.colors import ListedColormap

#Preformat figures
plt.rcParams['figure.dpi'] = 300
plt.style.use(['science','no-latex'])

#%% Create a custom colourmap (magenta-black and green-black)

#RBG value maximum (exclusive) 
N_map=256

#RBG values for the colours in the colourmap 
Colour_top=[233,98,233]              
Colour_centre=[98,12,82]
Colour_bottom=[0,0,0]

#Centre position of the colourmap
N_mapC=int(N_map/2)

#RGBA colourmap template. Last column is for the alpha-value (here set to 1)
MagentaBlack=np.ones((N_map,4))        

#Creates the custom colourmap (MagentaBlack)
for i in range(3):
    MagentaBlack[0:N_mapC,i]=np.linspace(Colour_bottom[i]/N_map,Colour_centre[i]/N_map,N_mapC)   
    MagentaBlack[N_mapC:N_map,i]=np.linspace(Colour_centre[i]/N_map,Colour_top[i]/N_map,N_mapC)

MagentaBlack=ListedColormap(MagentaBlack)    

#RBG value maximum (exclusive) 
N_map=256

#RBG values for the colours in the colourmap 
Colour_top=[0,256,0]              
Colour_centre=[0,128,0]
Colour_bottom=[0,0,0]

#Centre position of the colourmap
N_mapC=int(N_map/2)

#RGBA colourmap template. Last column is for the alpha-value (here set to 1)
GreenBlack=np.ones((N_map,4))        

#Creates the custom colourmap (MagentaBlack)
for i in range(3):
    GreenBlack[0:N_mapC,i]=np.linspace(Colour_bottom[i]/N_map,Colour_centre[i]/N_map,N_mapC)   
    GreenBlack[N_mapC:N_map,i]=np.linspace(Colour_centre[i]/N_map,Colour_top[i]/N_map,N_mapC)

GreenBlack=ListedColormap(GreenBlack)   

#%% Define global parameters 

#Binary interaction parameter
chi=3.0
#Interaction parameter and solvent order parameter at the critical point (first one is known, second one is set)
chi_c=2.0
phis_c=0.50

#Relative mobility of liquid (with respect to solvent)
M0=0.01
#Relative mobility of the colloids
Mc=M0/100

#Simulation timestep 
dt=0.50

#Gradient energy parameter 
kappa=0.50

#Attachment parameter 
alpha=30

#Solvent boundary condition 
BC=0

#%% 'Numbafied functions' used for solving the dynamic equations 

#Method that calculates the Laplacian of an input field 'F' via central finite-difference (5-point stencil)
@njit(parallel=True)
def Calc_Laplacian(F):               
    
    #Dimensions of the input field 
    Ny,Nx=F.shape
    
    #Create templates for the Laplacian contributions  
    Laplacian_x=np.zeros(F.shape)
    Laplacian_y=np.zeros(F.shape)
    
    #Calculate Laplacian 
    
    for j in prange(0,Ny,1):
        for i in prange(0,Nx,1):
            
            #Periodic boundary conditions
            L=(i-1)%Nx
            R=(i+1)%Nx
            B=(j+1)%Ny
            T=(j-1)%Ny
            
            #zero-flux boundary conditions
            T=j-1
            B=j+1
            
            if T==-1:
                T=j
            if B==Ny:
                B=j
            
            Laplacian_x[j,i]=-2*F[j,i]+F[j,R]+F[j,L]
            Laplacian_y[j,i]=-2*F[j,i]+F[B,i]+F[T,i]
                          
    Laplacian=Laplacian_x+Laplacian_y

    return Laplacian

#Method that calculates the 1D Laplacian of the solvent field, which uses different (flux) boundary conditions. The flux-boundary condition is set up with the BC Dirichlet condition. 
@njit(parallel=True)
def Calc_Laplacian_Solvent(F):               
    
    #Dimensions of the input field 
    Nz=len(F)
    
    #Create templates for the Laplacian contributions  
    Laplacian_z=np.zeros(Nz)
    
    #Calculate Laplacian 
    
    for k in prange(0,Nz,1):
        
        #zero-flux boundary condition
        D=k+1
        if D==Nz:
            D=k
        #flux boundary condition
        U=k-1
        if U==-1:
            Laplacian_z[k]=-2*F[k]+F[D]+BC
        else: 
            Laplacian_z[k]=-2*F[k]+F[D]+F[U]   
        
    Laplacian=Laplacian_z
    
    return Laplacian

#Method that calculates the gradient vector for an input field 'F' via central finite-difference (5-point stencil)
@njit(parallel=True)
def Calc_Gradient(F):            
    
    #Dimensions of the input field 
    Ny,Nx=F.shape
    
    #Create templates for the gradient vector components  
    Gradient_x=np.zeros(F.shape)                                                     
    Gradient_y=np.zeros(F.shape)
    
    for j in prange(0,Ny,1):
        for i in prange(0,Nx,1):
            
            #Periodic boundary conditions
            L=(i-1)%Nx
            R=(i+1)%Nx

            #zero-flux boundary conditions
            T=j-1
            B=j+1
            
            if T==-1:
                T=j
            if B==Ny:
                B=j
            
            Gradient_x[j,i]=(F[j,R]-F[j,L])/2
            Gradient_y[j,i]=(F[B,i]-F[T,i])/2
  
    Gradient=(Gradient_x,Gradient_y)
    
    return Gradient
    
#Method that calculates the solvent-dependent interaction parameter (linear)
@njit(parallel=True)
def Calc_Chi(phis):
    
    Nz=len(phis)
    Chi=np.zeros((Nz,Nz))
    
    for j in prange(Nz):
        for i in prange(Nz):
            Chi[j,i]=chi-(chi-chi_c)*phis[j]/phis_c
                
    return Chi

#Method that evolves the order parameter field according to the nondimensionalised dynamic equations
@njit(parallel=True)
def Propagate(phi,phis,psi,psic,n):
       
    #Calculate the oil-water interaction parameter depending on the solvent 
    Chi=Calc_Chi(phis)
    
    #Calculates the functional derivative of the free energy (chemical potential) of the liquids
    Mu=np.log(phi/(1-phi))+Chi*(1-2*phi)-kappa*Calc_Laplacian(phi)
    
    #Calculate mobility
    M=M0*1/2*(1-np.tanh((psi-psic)*n))
    
    #Update the order parameter field (Euler Forward)
    phi+=(M*Calc_Laplacian(Mu))*dt

    return phi

#Method that propagates the solvent order parameter field in time according to Fick's second law (nondimensionalised)
@njit(parallel=True)
def Propagate_Solvent(phi):
    
    #Update the order parameter field (Euler Forward)
    phi+=(Calc_Laplacian_Solvent(phi))*dt

    return phi
    
#Method that propagates the colloid order parameter field in time according to the nondimensionalised dynamic equations
@njit(parallel=True)
def Propagate_C(psi,phi,psic,n):
    
    #Calculate the gradient of the liquid field 
    Gradx,Grady=Calc_Gradient(phi)
    
    #Calculates the functional derivative of the free energy (chemical potential) for the colloids
    Mu=np.log(psi)-alpha/2*(Gradx**2+Grady**2)
    
    #Calculate the mobility of the colloids 
    MC=Mc*1/2*(1-np.tanh((psi-psic)*n))
    
    #Update order parameter field (Euler Forward)     
    psi+=(MC*Calc_Laplacian(Mu))*dt

    return psi

#%% Bare classes stripped of their Methods 

#The 'Liquid' class creates a PF object that evolves in accordance with a nondimensionalised dynamic equations
class Liquid:
    
    #Initialisation method that creates a field  (Size[0]xSize[1]) of composition phi0, including some random thermal noise
    def __init__(self,Size,phi0):
        self.Size=Size
        self.phi0=phi0
        self.phi=self.phi0+np.random.randint(-10,10,(self.Size))*0.001
        
#The 'Solvent' class creates a PF object that evolves in accordance with Fick's second law  
class Solvent:
    
    #Initialisation method that creates a field  (Size[0]) of composition phi0. 
    def __init__(self,Size,phi0):
  
        #Initialise liquid field 
        self.Size=Size
        self.phi0=phi0
        self.phi=np.zeros(self.Size)+self.phi0      
    

#The 'Colloid' class creates a PF object that is attracted to liquid-liquid interfaces and modulates the mobility 
class Colloid(Liquid):
     
    #Initisalisation method for the Colloid field (child class of 'Liquid')
    def __init__(self,Size,psi0):
        
        #Initialise parent class (Liquid)
        Liquid.__init__(self,Size,phi0)
        
        #Initialise colloid field 
        self.psi0=psi0
        self.psi=self.psi0+np.zeros(self.Size)
        
    
#%% Run phase-field simulation 

#Simulation dimensions 
Dimensions=(256,256)

#Initial liquid composition
phi0=0.50
#Initial solvent composition
phis0=0.50
#Initial colloid composition
psi0=0.50

#Threshold value for colloid jamming
psi_c=0.60
#''Sharpness'' of colloid mobility function
s=50

#Simulated time
t_sim=20000

#Total number of simulation steps 
N=int(t_sim/dt)    

#Initialise the liquid order parameter field 
Oil=Liquid(Dimensions,phi0)

#Initialise the solvent order parameter field 
Alchohol=Solvent(Dimensions[0],phis0)

#Initialise the colloid field 
Colloids=Colloid(Dimensions,psi0)

#Start-point for determining computation time 
t0=perf_counter()

for n in range(N+1):
    
    #Create local copies of the liquid and colloid field 
    phi,phis,psi=np.copy(Oil.phi),np.copy(Alchohol.phi),np.copy(Colloids.psi)

    #Propagate the fields in time via the Dynamic equations/Fick's second law
    Oil.phi=Propagate(phi,phis,psi,psi_c,s)
    Alchohol.phi=Propagate_Solvent(phis)
    Colloids.psi=Propagate_C(psi,phi,psi_c,s)
    
    #Check progress at 5% intervals   
    if n%(N/20)==0:
        print(str(100*n/N)+'% complete')

#End-point for determining computation time
t1=perf_counter()   
#Total computation time
Deltat=t1-t0         
print('Computation time: '+str(Deltat)+' s') 

#%% Visualise final state

plt.figure()
plt.xlabel(r'Horizontal position, $\tilde{x}$')
plt.ylabel(r'Vertical position, $\tilde{y}$')
plt.title(r'Liquids, $\phi$')    
plt.imshow(Oil.phi,norm=clr.Normalize(vmin=0.0707,vmax=1-0.0707),cmap=MagentaBlack)
plt.minorticks_on()
plt.colorbar()

plt.figure()
plt.xlabel(r'Horizontal position, $\tilde{x}$',color='white')
plt.ylabel(r'Vertical position, $\tilde{y}$',color='white')
plt.title(r'Nanoparticles, $\psi$')
plt.imshow(Colloids.psi,vmin=0,vmax=0.6,cmap=GreenBlack)
plt.minorticks_on()
plt.colorbar()

