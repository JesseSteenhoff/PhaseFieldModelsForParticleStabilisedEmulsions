# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 10:52:34 2026

@author: J.M. Steenhoff
"""
#Import all the required modules 
import numpy as np
from numba import njit,prange
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import scienceplots

from time import perf_counter
from scipy.optimize import fsolve
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
#Simulation timestep 
dt=0.01
#Relative mobility of liquid
M0=1
#Relative mobility of the colloids
Mc=M0/100
#Attachment parameter 
alpha=60

#%% 'Numbafied functions' used for solving the dynamic equations 

#Method that evolves the order parameter field according to the nondimensionalised dynamic equations
@njit(parallel=True,fastmath=True)
def Propagate(phi,psi,psic,n,Mu,M,M_faces_x,M_faces_y,Jx,Jy):
    
    #Lattice dimensions 
    Ny,Nx=phi.shape
    
    for j in prange(Ny):
        #Periodic boundary conditions
        B=(j+1)%Ny
        T=(j-1)%Ny
        for i in prange(Nx):
            
            #Periodic boundary conditions
            L=(i-1)%Nx
            R=(i+1)%Nx
           
            #Calculate Laplacian 
            Laplacian=-4*phi[j,i]+phi[j,R]+phi[j,L]+phi[B,i]+phi[T,i]
            #Calculate functional derivative of free-energy functional / chemical potential 
            Mu[j,i]=np.log(phi[j,i]/(1-phi[j,i]))+chi*(1-2*phi[j,i])-Laplacian
            
            #Calculate Mobility 
            M[j,i]=M0*1/2*(1-np.tanh((psi[j,i]-psic)*n))
    
    # Calculate mobility at lattice faces
    for j in prange(Ny):
        for i in prange(1,Nx):
            M_faces_x[j,i]=2*M[j,i]*M[j,i-1]/(M[j,i]+M[j,i-1])
    
    for j in prange(1,Ny):
        for i in prange(Nx):
            M_faces_y[j,i]=2*M[j,i]*M[j-1,i]/(M[j,i]+M[j-1,i])
            
    #Calculate fluxes at lattice faces
    for j in prange(Ny):
        for i in prange(1,Nx):
            Jx[j,i]=-M_faces_x[j,i]*(Mu[j,i]-Mu[j,i-1])
    
    for j in prange(1,Ny):
        for i in prange(Nx):
            Jy[j,i]=-M_faces_y[j,i]*(Mu[j,i]-Mu[j-1,i])
   
    #Impose periodic boundary conditions
    for j in prange(Ny):
        M_faces_x[j,0]=2*M[j,0]*M[j,Nx-1]/(M[j,0]+M[j,Nx-1])
        Jx[j,0]=-M_faces_x[j,0]*(Mu[j,0]-Mu[j,Nx-1])
        Jx[j,Nx]=Jx[j,0]
    for i in prange(Nx):
        M_faces_y[0,i]=2*M[0,i]*M[Ny-1,i]/(M[0,i]+M[Ny-1,i])
        Jy[0,i]=-M_faces_y[0,i]*(Mu[0,i]-Mu[Ny-1,i])
        Jy[Ny,i]=Jy[0,i]
    
    for j in prange(Ny):
        for i in prange(Nx):
            #Approximate divergence of fluxes at lattice points (finite-difference)
            DivJ_x=(Jx[j,i+1]-Jx[j,i])
            DivJ_y=(Jy[j+1,i]-Jy[j,i])
            # Update order parameter field (Euler Forward)
            phi[j,i]-=(DivJ_x+DivJ_y)*dt
            
    return phi
    
#Method that propagates the colloid order parameter field in time according to the nondimensionalised dynamic equations
@njit(parallel=True,fastmath=True)
def Propagate_C(psi,phi,psic,n,Mu,MC,MC_faces_x,MC_faces_y,Jx,Jy):
    
    #Lattice dimensions 
    Ny,Nx=psi.shape
    
    for j in prange(Ny):
        #Periodic boundary conditions
        B=(j+1)%Ny
        T=(j-1)%Ny
        for i in prange(Nx):
            
            #Periodic boundary conditions
            L=(i-1)%Nx
            R=(i+1)%Nx
            
            #Calculate Gradient 
            Gradx=(phi[j,R]-phi[j,L])/2
            Grady=(phi[B,i]-phi[T,i])/2
            
            #Calculate functional derivative of free-energy functional / chemical potential 
            Mu[j,i]=np.log(psi[j,i])-alpha/2*(Gradx*Gradx+Grady*Grady)
            
            #Calculate Mobility 
            MC[j,i]=Mc*1/2*(1-np.tanh((psi[j,i]-psic)*n))
            
    # Calculate mobility at lattice faces
    for j in prange(Ny):
        for i in prange(1,Nx):
            MC_faces_x[j,i]=2*MC[j,i]*MC[j,i-1]/(MC[j,i]+MC[j,i-1])
    
    for j in prange(1,Ny):
        for i in prange(Nx):
            MC_faces_y[j,i]=2*MC[j,i]*MC[j-1,i]/(MC[j,i]+MC[j-1,i])
            
    #Calculate fluxes at lattice faces
    for j in prange(Ny):
        for i in prange(1,Nx):
            Jx[j,i]=-MC_faces_x[j,i]*(Mu[j,i]-Mu[j,i-1])
    
    for j in prange(1,Ny):
        for i in prange(Nx):
            Jy[j,i]=-MC_faces_y[j,i]*(Mu[j,i]-Mu[j-1,i])
   
    #Impose periodic boundary conditions
    for j in prange(Ny):
        MC_faces_x[j,0]=2*MC[j,0]*MC[j,Nx-1]/(MC[j,0]+MC[j,Nx-1])
        Jx[j,0]=-MC_faces_x[j,0]*(Mu[j,0]-Mu[j,Nx-1])
        Jx[j,Nx]=Jx[j,0]
    for i in prange(Nx):
        MC_faces_y[0,i]=2*MC[0,i]*MC[Ny-1,i]/(MC[0,i]+MC[Ny-1,i])
        Jy[0,i]=-MC_faces_y[0,i]*(Mu[0,i]-Mu[Ny-1,i])
        Jy[Ny,i]=Jy[0,i]
    
    for j in prange(Ny):
        for i in prange(Nx):
            #Approximate divergence of fluxes at lattice points (finite-difference)
            DivJ_x=(Jx[j,i+1]-Jx[j,i])
            DivJ_y=(Jy[j+1,i]-Jy[j,i])
            # Update order parameter field (Euler Forward)
            psi[j,i]-=(DivJ_x+DivJ_y)*dt
        
    return psi
    
    
#%% Bare classes stripped of their Methods 

#The 'Liquid' class creates a PF object that evolves in accordance with a nondimensionalised dynamic equations
class Liquid:
    
    #Initialisation method that creates a field  (Size[0]xSize[1]) of composition phi0, including some random thermal noise
    def __init__(self,Size,phi0):
        self.Size=Size
        self.phi0=phi0
        self.phi=self.phi0+np.random.randint(-10,10,(self.Size))*0.001
        
        #Solve the binodal equation to find the equilibrium compositions 
        self.phimax=fsolve(lambda x: np.log(x/(1-x))+chi*(1-2*x),0.99)
        self.phimin=1-self.phimax
        
    #Method that visualises current state of the phase-field (liquid)
    def Show(self,C):                           
        plt.figure()
        plt.xlabel(r'Horizontal position, $\tilde{x}$')
        plt.ylabel(r'Vertical position, $\tilde{y}$')
        plt.imshow(self.phi,norm=clr.Normalize(vmin=self.phimin,vmax=self.phimax),cmap=C)
        plt.colorbar()
        return None
  
#The 'Colloid' class creates a PF object that is attracted to liquid-liquid interfaces and modulates the mobility 
class Colloid(Liquid):
     
    #Initisalisation method for the Colloid field (child class of 'Liquid')
    def __init__(self,Size,psi0):
        
        #Initialise parent class (Liquid)
        Liquid.__init__(self,Size,phi0)
        
        #Initialise colloid field 
        self.psi0=psi0
        self.psi=self.psi0+np.zeros(self.Size)
    
    #Method that visualises the current state of the colloid phase-field
    def Show(self):                           
        plt.figure()
        plt.xlabel(r'Horizontal position, $\tilde{x}$')
        plt.ylabel(r'Vertical position, $\tilde$')
        plt.imshow(self.psi,norm=clr.Normalize(vmin=0,vmax=1),cmap=GreenBlack)
        plt.colorbar()
        return None
    
#%% Run phase-field simulation 

#Simulation dimensions 
Dimensions=(128,128)

#Initial composition (liquids, colloids)
phi0=0.50
psi0=0.50
#Threshold value for colloid jamming
psi_c=0.60
#''Sharpness'' of colloid mobility function
s=75

#Simulated time
t_sim=100

#Total number of simulation steps 
N=int(t_sim/dt)    

#Initialise the liquid order parameter field 
Oil=Liquid(Dimensions,phi0)

#Initialise the colloid field 
Colloids=Colloid(Dimensions,psi0)         

#Storage lists for the morphologies 
Liquid_Morphology=[]
Colloid_Morphology=[]

#Storage list for the mobility
M_list=[]
#Storage list for the average nanoparticle content 
psi_av=[]
#Storage list for the average liquid 
phi_av=[]

#Set-up working arrays 
Mu=np.zeros(Dimensions)
Jx=np.zeros((Dimensions[0],Dimensions[1]+1))
Jy=np.zeros((Dimensions[0]+1,Dimensions[1]))
M_faces_x=np.zeros((Dimensions[0],Dimensions[1]+1))
M_faces_y=np.zeros((Dimensions[0]+1,Dimensions[1]))
M=np.zeros(Dimensions)

#Start-point for determining computation time 
t0=perf_counter()

for n in range(N):

    #Create local copies of the liquid and colloid field 
    phi,psi=np.copy(Oil.phi),np.copy(Colloids.psi)

    #Propagate the fields in time via the dynamic equations for jamming particles
    Oil.phi=Propagate(phi,psi,psi_c,s,Mu,M,M_faces_x,M_faces_y,Jx,Jy)
    Colloids.psi=Propagate_C(psi,phi,psi_c,s,Mu,M,M_faces_x,M_faces_y,Jx,Jy)
    
    #Export system morphology at 1% intervals
    if ((n+1)*dt)%(t_sim/100)==0:
        Liquid_Morphology.append(np.copy(Oil.phi))
        Colloid_Morphology.append(np.copy(Colloids.psi))
        M_list.append(M0*1/2*(1-np.tanh((np.copy(Colloids.psi)-psi_c)*s)))
        psi_av.append(np.sum(np.copy(Colloids.psi))/(Dimensions[0]*Dimensions[1]))
        phi_av.append(np.sum(np.copy(Oil.phi))/(Dimensions[0]*Dimensions[1]))
    
    #Check progress at 10% intervals
    if ((n+1)*dt)%(t_sim/10)==0:
        Oil.Show(MagentaBlack)
        print(str(100*(n+1)/N)+'% complete')
        
#End-point for determining computation time
t1=perf_counter()   
#Total computation time
Deltat=t1-t0         
print('Computation time: '+str(Deltat)+' s')

#%% Track average value of the nanoparticle and liquid content over time

#Calculate time-range
t_range=np.arange(t_sim/100,t_sim+t_sim/100,t_sim/100)

plt.figure()
plt.minorticks_on()
plt.xlabel(r'Simulated time, $\tilde{t}$')
plt.ylabel('Av. Component Content')

plt.plot(t_range,psi_av,label=r'$\psi_{av}$')
plt.plot(t_range,phi_av,label=r'$\phi_{av}$')

plt.legend()

#%% Visualise the morphologies

#Select Morphology
Index=60
#Selected Time
t=t_range[Index]
#In percentages
Perc=(Index+1)

plt.figure()
plt.xlabel(r'Horizontal position, $\tilde{x}$')
plt.ylabel(r'Vertical position, $\tilde{y}$')
plt.title(r'Liquids, $\phi$')    
plt.imshow(Liquid_Morphology[Index],norm=clr.Normalize(vmin=Oil.phimin,vmax=Oil.phimax),cmap=MagentaBlack)
plt.minorticks_on()
plt.colorbar()


plt.figure()
plt.xlabel(r'Horizontal position, $\tilde{x}$',color='white')
plt.ylabel(r'Vertical position, $\tilde{y}$',color='white')
plt.title(r'Nanoparticles, $\psi$')
#plt.tick_params(colors='white', which='both')
plt.imshow(Colloid_Morphology[Index],vmin=np.min(Colloid_Morphology),vmax=np.max(Colloid_Morphology),cmap=GreenBlack)
plt.minorticks_on()
plt.colorbar()


plt.figure()
plt.xlabel(r'Horizontal position, $\tilde{x}$',color='white')
plt.ylabel(r'Vertical position, $\tilde{y}$',color='white')
plt.title(r'General mobility, $\tilde{M}_i/\tilde{M}_i^0$')    
plt.imshow(M_list[Index],vmin=0,vmax=1,cmap='binary_r')

#plt.tick_params(colors='white', which='both')
plt.minorticks_on()
plt.colorbar()

