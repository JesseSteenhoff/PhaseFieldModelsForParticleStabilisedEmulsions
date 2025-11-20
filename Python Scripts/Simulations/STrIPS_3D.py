# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 19:35:03 2025

@author: J.M. Steenhoff
"""
#Import all the required modules 
import numpy as np
from numba import njit,prange
from time import perf_counter

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

#Method that calculates the Laplacian of an input field 'F' via central finite-difference (7-point stencil)
@njit(parallel=True)
def Calc_Laplacian(F):               
    
    #Dimensions of the input field 
    Nz,Ny,Nx=F.shape
    
    #Create templates for the Laplacian contributions  
    Laplacian_x=np.zeros(F.shape)
    Laplacian_y=np.zeros(F.shape)
    Laplacian_z=np.zeros(F.shape)
    
    #Calculate Laplacian 
    
    for k in prange(0,Nz,1):
        for j in prange(0,Ny,1):
            for i in prange(0,Nx,1):
                
                #Periodic boundary conditions
                L=(i-1)%Nx
                R=(i+1)%Nx
                B=(j+1)%Ny
                T=(j-1)%Ny
                
                #zero-flux boundary conditions
                U=k-1
                D=k+1
                
                if U==-1:
                    U=k
                if D==Nz:
                    D=k
                
                Laplacian_x[k,j,i]=-2*F[k,j,i]+F[k,j,R]+F[k,j,L]
                Laplacian_y[k,j,i]=-2*F[k,j,i]+F[k,B,i]+F[k,T,i]
                Laplacian_z[k,j,i]=-2*F[k,j,i]+F[D,j,i]+F[U,j,i]                    
  
    Laplacian=Laplacian_x+Laplacian_y+Laplacian_z

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

#Method that calculates the gradient vector for an input field 'F' via central finite-difference (7-point stencil)
@njit(parallel=True)
def Calc_Gradient(F):            
    
    #Dimensions of the input field 
    Nz,Ny,Nx=F.shape
    
    #Create templates for the gradient vector components  
    Gradient_x=np.zeros(F.shape)                                                     
    Gradient_y=np.zeros(F.shape)
    Gradient_z=np.zeros(F.shape)
    
    for k in prange(0,Nz,1):
        for j in prange(0,Ny,1):
            for i in prange(0,Nx,1):
                
                #Periodic boundary conditions
                L=(i-1)%Nx
                R=(i+1)%Nx
                B=(j+1)%Ny
                T=(j-1)%Ny
                
                #zero-flux boundary conditions
                U=k-1
                D=k+1
                
                if U==-1:
                    U=k
                if D==Nz:
                    D=k
                
                Gradient_x[k,j,i]=(F[k,j,R]-F[k,j,L])/2
                Gradient_y[k,j,i]=(F[k,B,i]-F[k,T,i])/2
                Gradient_z[k,j,i]=(F[D,j,i]-F[U,j,i])/2       
    
    Gradient=(Gradient_x,Gradient_y,Gradient_z)
    
    return Gradient
    
#Method that calculates the solvent-dependent interaction parameter (linear)
@njit(parallel=True)
def Calc_Chi(phis):
    
    Nz=len(phis)
    Chi=np.zeros((Nz,Nz,Nz))
    
    for k in prange(Nz):
        for j in prange(Nz):
            for i in prange(Nz):
                Chi[k,j,i]=chi-(chi-chi_c)*phis[k]/phis_c
                
    return Chi

#Method that evolves the order parameter field according to the nondimensionalised dynamic equations 
@njit(parallel=True)
def Propagate(phi,phis,psi,psic,n):
       
    #Calculate the oil-water interaction parameter depending on the solvent 
    Chi=Calc_Chi(phis)
    
    #Calculates the functional derivative of the free energy (chemical potential) for the liquids
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
    Gradx,Grady,Gradz=Calc_Gradient(phi)
    
    #Calculates the functional derivative of the free energy (chemical potential)
    Mu=np.log(psi)-alpha/2*(Gradx**2+Grady**2+Gradz**2)
    
    #Calculate the mobility of the colloids 
    MC=Mc*1/2*(1-np.tanh((psi-psic)*n))
    
    #Update order parameter field (Euler Forward)     
    psi+=(MC*Calc_Laplacian(Mu))*dt

    return psi

#%% Bare classes stripped of their Methods 

#The 'Liquid' class creates a PF object that evolves in accordance with nondimensionalised dynamic equations
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
Dimensions=(256,256,256)

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
t_sim=30000

#Initialise the liquid order parameter field 
Oil=Liquid(Dimensions,phi0)

#Initialise the solvent order parameter field 
Alchohol=Solvent(Dimensions[0],phis0)

#Initialise the colloid field 
Colloids=Colloid(Dimensions,psi0)

#Total number of simulation steps 
N=int(t_sim/dt)                  

#Start-point for determining computation time 
t0=perf_counter()

for n in range(N+1):
    
    #Create local copies of the liquid and colloid field 
    phi,phis,psi=np.copy(Oil.phi),np.copy(Alchohol.phi),np.copy(Colloids.psi)

    #Propagate the fields in time via the Cahn-Hilliard equation/Fick's second law
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
