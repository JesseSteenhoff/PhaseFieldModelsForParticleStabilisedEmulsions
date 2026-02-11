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
#Interaction parameter and solvent order parameter at the critical point (first one is known, second one is set)
chi_c=2.0
phis_c=0.50
#Simulation timestep 
dt=0.50
#Relative mobility of liquid (relative to solvent)
M0=0.01
#Relative mobility of the colloids (relative to solvent)
Mc=M0/100
#Attachment parameter 
alpha=60
#Solvent Boundary Condition 
BC=0

#%% 'Numbafied functions' used for solving the dynamic equations 

#Method that evolves the liquid order parameter field according to the nondimensionalised dynamic equations
@njit(parallel=True,fastmath=True)
def Propagate(phi,psi,phis,psic,n,Mu,M,M_faces_x,M_faces_y,M_faces_z,Jx,Jy,Jz):
    
    #Lattice dimensions 
    Nz,Ny,Nx=phi.shape
    
    for k in prange(1,Nz-1):
        U=k-1
        D=k+1
        for j in prange(Ny):
            #Periodic boundary conditions
            B=(j+1)%Ny
            T=(j-1)%Ny
            for i in prange(Nx):
                
                #Periodic boundary conditions
                L=(i-1)%Nx
                R=(i+1)%Nx
            
                #Calculate Laplacian 
                Laplacian=-6*phi[k,j,i]+phi[k,j,R]+phi[k,j,L]+phi[k,B,i]+phi[k,T,i]+phi[U,j,i]+phi[D,j,i]
                #Calculate solvent-dependent interaction parameter 
                Chi=chi-(chi-chi_c)*phis[k]/phis_c
                #Calculate functional derivative of free-energy functional / chemical potential 
                Mu[k,j,i]=np.log(phi[k,j,i]/(1-phi[k,j,i]))+Chi*(1-2*phi[k,j,i])-Laplacian
                
                #Calculate Mobility 
                M[k,j,i]=M0*1/2*(1-np.tanh((psi[k,j,i]-psic)*n))
    
    #No-Flux Boundary Conditions of Top Row 
    for j in prange(Ny):
        for i in prange(Nx):
            
            U=0
            D=1
            
            #Periodic boundary conditions
            B=(j+1)%Ny
            T=(j-1)%Ny
            L=(i-1)%Nx
            R=(i+1)%Nx
           
            #Calculate Laplacian 
            Laplacian=-6*phi[0,j,i]+phi[0,j,R]+phi[0,j,L]+phi[0,B,i]+phi[0,T,i]+phi[U,j,i]+phi[D,j,i]
            #Calculate solvent-dependent interaction parameter 
            Chi=chi-(chi-chi_c)*phis[0]/phis_c
            #Calculate functional derivative of free-energy functional / chemical potential 
            Mu[0,j,i]=np.log(phi[0,j,i]/(1-phi[0,j,i]))+Chi*(1-2*phi[0,j,i])-Laplacian
        
            #Calculate Mobility 
            M[0,j,i]=M0*1/2*(1-np.tanh((psi[0,j,i]-psic)*n))
    
    #No-Flux Boundary Conditions of Bottom Row 
    for j in prange(Ny):
        for i in prange(Nx):
            
            U=Nz-2
            D=Nz-1
            
            #Periodic boundary conditions
            B=(j+1)%Ny
            T=(j-1)%Ny
            L=(i-1)%Nx
            R=(i+1)%Nx
           
            #Calculate Laplacian 
            Laplacian=-6*phi[Nz-1,j,i]+phi[Nz-1,j,R]+phi[Nz-1,j,L]+phi[Nz-1,B,i]+phi[Nz-1,T,i]+phi[U,j,i]+phi[D,j,i]
            #Calculate solvent-dependent interaction parameter 
            Chi=chi-(chi-chi_c)*phis[Nz-1]/phis_c
            #Calculate functional derivative of free-energy functional / chemical potential 
            Mu[Nz-1,j,i]=np.log(phi[Nz-1,j,i]/(1-phi[Nz-1,j,i]))+Chi*(1-2*phi[Nz-1,j,i])-Laplacian
        
            #Calculate Mobility 
            M[Nz-1,j,i]=M0*1/2*(1-np.tanh((psi[Nz-1,j,i]-psic)*n))
    
     
    # Calculate mobility at lattice faces
    for k in prange(Nz):
        for j in prange(Ny):
            for i in prange(1,Nx):
                M_faces_x[k,j,i]=2*M[k,j,i]*M[k,j,i-1]/(M[k,j,i]+M[k,j,i-1])
    
    for k in prange(Nz):
        for j in prange(1,Ny):
            for i in prange(Nx):
                M_faces_y[k,j,i]=2*M[k,j,i]*M[k,j-1,i]/(M[k,j,i]+M[k,j-1,i])
    
    for k in prange(1,Nz):
        for j in prange(Ny):
            for i in prange(Nx):
                M_faces_z[k,j,i]=2*M[k,j,i]*M[k-1,j,i]/(M[k,j,i]+M[k-1,j,i])
    
    #Calculate fluxes at lattice faces
    
    for k in prange(Nz):
        for j in prange(Ny):
            for i in prange(1,Nx):
                Jx[k,j,i]=-M_faces_x[k,j,i]*(Mu[k,j,i]-Mu[k,j,i-1])
    
    for k in prange(Nz):
        for j in prange(1,Ny):
            for i in prange(Nx):
                Jy[k,j,i]=-M_faces_y[k,j,i]*(Mu[k,j,i]-Mu[k,j-1,i])
   
    for k in prange(1,Nz):
        for j in prange(Ny):
            for i in prange(Nx):
                Jz[k,j,i]=-M_faces_z[k,j,i]*(Mu[k,j,i]-Mu[k-1,j,i])
    
   
    #Impose periodic boundary conditions
    for k in prange(Nz):
        for j in prange(Ny):
            M_faces_x[k,j,0]=2*M[k,j,0]*M[k,j,Nx-1]/(M[k,j,0]+M[k,j,Nx-1])
            Jx[k,j,0]=-M_faces_x[k,j,0]*(Mu[k,j,0]-Mu[k,j,Nx-1])
            Jx[k,j,Nx]=Jx[k,j,0]
        for i in prange(Nx):
            M_faces_y[k,0,i]=2*M[k,0,i]*M[k,Ny-1,i]/(M[k,0,i]+M[k,Ny-1,i])
            Jy[k,0,i]=-M_faces_y[k,0,i]*(Mu[k,0,i]-Mu[k,Ny-1,i])
            Jy[k,Ny,i]=Jy[k,0,i]
    #Impose No-Flux boundary conditions
    for j in prange(Ny):
        for i in prange(Nx):
            Jz[0,j,i]=0
            Jz[Nz,j,i]=0
   
    for k in prange(Nz):
        for j in prange(Ny):
            for i in prange(Nx):
                #Approximate divergence of fluxes at lattice points (finite-difference)
                DivJ_x=(Jx[k,j,i+1]-Jx[k,j,i])
                DivJ_y=(Jy[k,j+1,i]-Jy[k,j,i])
                DivJ_z=(Jz[k+1,j,i]-Jz[k,j,i])
                # Update order parameter field (Euler Forward)
                phi[k,j,i]-=(DivJ_x+DivJ_y+DivJ_z)*dt
            
    return phi
    
#Method that propagates the colloid order parameter field in time according to the nondimensionalised dynamic equations
@njit(parallel=True,fastmath=True)
def Propagate_C(psi,phi,psic,n,Mu,MC,MC_faces_x,MC_faces_y,MC_faces_z,Jx,Jy,Jz):
    
    #Lattice dimensions 
    Nz,Ny,Nx=psi.shape
    
    for k in prange(1,Nz-1):
        U=k-1
        D=k+1
        for j in prange(Ny):
            #Periodic boundary conditions
            B=(j+1)%Ny
            T=(j-1)%Ny
            for i in prange(Nx):
                
                #Periodic boundary conditions
                L=(i-1)%Nx
                R=(i+1)%Nx
                
                #Calculate Gradient 
                Gradx=(phi[k,j,R]-phi[k,j,L])/2
                Grady=(phi[k,B,i]-phi[k,T,i])/2
                Gradz=(phi[D,j,i]-phi[U,j,i])/2
            
                #Calculate functional derivative of free-energy functional / chemical potential 
                Mu[k,j,i]=np.log(psi[k,j,i])-alpha/2*(Gradx*Gradx+Grady*Grady+Gradz*Gradz)
                                
                #Calculate Mobility 
                MC[k,j,i]=Mc*1/2*(1-np.tanh((psi[k,j,i]-psic)*n))
    
    #No-Flux Boundary Conditions of Top Row 
    for j in prange(Ny):
        for i in prange(Nx):
            
            U=0
            D=1
            
            #Periodic boundary conditions
            B=(j+1)%Ny
            T=(j-1)%Ny
            L=(i-1)%Nx
            R=(i+1)%Nx
            
            #Calculate Gradient 
            Gradx=(phi[0,j,R]-phi[0,j,L])/2
            Grady=(phi[0,B,i]-phi[0,T,i])/2
            Gradz=(phi[D,j,i]-phi[U,j,i])/2

            #Calculate functional derivative of free-energy functional / chemical potential 
            Mu[0,j,i]=np.log(psi[0,j,i])-alpha/2*(Gradx*Gradx+Grady*Grady+Gradz*Gradz)
                            
            #Calculate Mobility 
            MC[0,j,i]=Mc*1/2*(1-np.tanh((psi[0,j,i]-psic)*n))
    
    #No-Flux Boundary Conditions of Bottom Row 
    for j in prange(Ny):
        for i in prange(Nx):
            
            U=Nz-2
            D=Nz-1
            
            #Periodic boundary conditions
            B=(j+1)%Ny
            T=(j-1)%Ny
            L=(i-1)%Nx
            R=(i+1)%Nx
           
            #Calculate Gradient 
            Gradx=(phi[Nz-1,j,R]-phi[Nz-1,j,L])/2
            Grady=(phi[Nz-1,B,i]-phi[Nz-1,T,i])/2
            Gradz=(phi[D,j,i]-phi[U,j,i])/2

            #Calculate functional derivative of free-energy functional / chemical potential 
            Mu[Nz-1,j,i]=np.log(psi[Nz-1,j,i])-alpha/2*(Gradx*Gradx+Grady*Grady+Gradz*Gradz)
                            
            #Calculate Mobility 
            MC[Nz-1,j,i]=Mc*1/2*(1-np.tanh((psi[Nz-1,j,i]-psic)*n))
    
     
    # Calculate mobility at lattice faces
    for k in prange(Nz):
        for j in prange(Ny):
            for i in prange(1,Nx):
                MC_faces_x[k,j,i]=2*MC[k,j,i]*MC[k,j,i-1]/(MC[k,j,i]+MC[k,j,i-1])
    
    for k in prange(Nz):
        for j in prange(1,Ny):
            for i in prange(Nx):
                MC_faces_y[k,j,i]=2*MC[k,j,i]*MC[k,j-1,i]/(MC[k,j,i]+MC[k,j-1,i])
    
    for k in prange(1,Nz):
        for j in prange(Ny):
            for i in prange(Nx):
                MC_faces_z[k,j,i]=2*MC[k,j,i]*MC[k-1,j,i]/(MC[k,j,i]+MC[k-1,j,i])
    
    #Calculate fluxes at lattice faces
    for k in prange(Nz):
        for j in prange(Ny):
            for i in prange(1,Nx):
                Jx[k,j,i]=-MC_faces_x[k,j,i]*(Mu[k,j,i]-Mu[k,j,i-1])
    
    for k in prange(Nz):
        for j in prange(1,Ny):
            for i in prange(Nx):
                Jy[k,j,i]=-MC_faces_y[k,j,i]*(Mu[k,j,i]-Mu[k,j-1,i])
   
    for k in prange(1,Nz):
        for j in prange(Ny):
            for i in prange(Nx):
                Jz[k,j,i]=-MC_faces_z[k,j,i]*(Mu[k,j,i]-Mu[k-1,j,i])
    
   
    #Impose periodic boundary conditions
    for k in prange(Nz):
        for j in prange(Ny):
            MC_faces_x[k,j,0]=2*MC[k,j,0]*MC[k,j,Nx-1]/(MC[k,j,0]+MC[k,j,Nx-1])
            Jx[k,j,0]=-MC_faces_x[k,j,0]*(Mu[k,j,0]-Mu[k,j,Nx-1])
            Jx[k,j,Nx]=Jx[k,j,0]
        for i in prange(Nx):
            MC_faces_y[k,0,i]=2*MC[k,0,i]*MC[k,Ny-1,i]/(MC[k,0,i]+MC[k,Ny-1,i])
            Jy[k,0,i]=-MC_faces_y[k,0,i]*(Mu[k,0,i]-Mu[k,Ny-1,i])
            Jy[k,Ny,i]=Jy[k,0,i]
    #Impose No-Flux boundary conditions
    for j in prange(Ny):
        for i in prange(Nx):
            Jz[0,j,i]=0
            Jz[Nz,j,i]=0
   
    for k in prange(Nz):
        for j in prange(Ny):
            for i in prange(Nx):
                #Approximate divergence of fluxes at lattice points (finite-difference)
                DivJ_x=(Jx[k,j,i+1]-Jx[k,j,i])
                DivJ_y=(Jy[k,j+1,i]-Jy[k,j,i])
                DivJ_z=(Jz[k+1,j,i]-Jz[k,j,i])
                # Update order parameter field (Euler Forward)
                psi[k,j,i]-=(DivJ_x+DivJ_y+DivJ_z)*dt
        
    return psi

#Method that evolves the solvent order parameter field according to the nondimensionalised Fick's second law. This method solves the equation in 1D. 
def Propagate_Solvent(phis):
    
    #Lattice dimensions 
    Ny=len(phis)
    
    for j in prange(Ny):
        #No-Flux boundary condition
        B=j+1
        if B==Ny:
            B=j
        #Flux boundary condition
        T=j-1
        if T==-1:
            Laplacian=-2*phis[j]+phis[B]+BC
        else: 
            Laplacian=-2*phis[j]+phis[B]+phis[T]
        
        phis[j]+=Laplacian*dt
        
    return phis
    
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
    def Show(self,Index,C):                           
        plt.figure()
        plt.xlabel(r'Horizontal position, $\tilde{x}$')
        plt.ylabel(r'Vertical position, $\tilde{y}$')
        plt.imshow(self.phi[Index],norm=clr.Normalize(vmin=self.phimin,vmax=self.phimax),cmap=C)
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
    def Show(self,Index):                           
        plt.figure()
        plt.imshow(self.psi[Index],norm=clr.Normalize(vmin=0,vmax=1),cmap=GreenBlack)
        plt.colorbar()
        return None

#The 'Solvent' class creates a PF object that evolves in accordance with Fick's second law  
class Solvent:
    
    #Initialisation method that creates a field  (Size[0]) of composition phi0. 
    def __init__(self,Size,phis0):
  
        #Initialise liquid field 
        self.Size=Size
        self.phis0=phis0
        self.phis=np.zeros(self.Size)+self.phis0      
    
    
#%% Run phase-field simulation 

#Simulation dimensions 
Dimensions=(256,256,256)

#Initial composition (liquids, colloids,solvent)
phi0=0.50
psi0=0.50
phis0=0.50

#Threshold value for colloid jamming
psi_c=0.60
#''Sharpness'' of colloid mobility function
s=50

#Simulated time
t_sim=30000

#Total number of simulation steps 
N=int(t_sim/dt)    

#Initialise the liquid order parameter field 
Oil=Liquid(Dimensions,phi0)

#Initialise the colloid field 
Colloids=Colloid(Dimensions,psi0)         

#Initialise the solvent order parameter field 
Alchohol=Solvent(Dimensions[0],phis0)

#Storage lists for the morphologies 
Liquid_Morphology=[]
Colloid_Morphology=[]
Alchohol_Profiles=[]

#Storage list for the mobility
M_list=[]
#Storage list for the average nanoparticle content 
psi_av=[]
#Storage list for the average liquid 
phi_av=[]

#Set-up working arrays 
Mu=np.zeros(Dimensions)
Jx=np.zeros((Dimensions[0],Dimensions[1],Dimensions[2]+1))
Jy=np.zeros((Dimensions[0],Dimensions[1]+1,Dimensions[2]))
Jz=np.zeros((Dimensions[0]+1,Dimensions[1],Dimensions[2]))
M_faces_x=np.zeros((Dimensions[0],Dimensions[1],Dimensions[2]+1))
M_faces_y=np.zeros((Dimensions[0],Dimensions[1]+1,Dimensions[2]))
M_faces_z=np.zeros((Dimensions[0]+1,Dimensions[1],Dimensions[2]))
M=np.zeros(Dimensions)

#Start-point for determining computation time 
t0=perf_counter()

for n in range(N):

    #Create local copies of the liquid and colloid field 
    phi,psi,phis=np.copy(Oil.phi),np.copy(Colloids.psi),np.copy(Alchohol.phis)

    #Propagate the fields in time via the dynamic equations for jamming particles / Fick's second law 
    Oil.phi=Propagate(phi,psi,phis,psi_c,s,Mu,M,M_faces_x,M_faces_y,M_faces_z,Jx,Jy,Jz)
    Colloids.psi=Propagate_C(psi,phi,psi_c,s,Mu,M,M_faces_x,M_faces_y,M_faces_z,Jx,Jy,Jz)
    Alchohol.phis=Propagate_Solvent(phis)
    
    #Export system morphology at 1% intervals
    if ((n+1)*dt)%(t_sim/100)==0:
        Liquid_Morphology.append(np.copy(Oil.phi))
        Colloid_Morphology.append(np.copy(Colloids.psi))
        Alchohol_Profiles.append(np.copy(Alchohol.phis))
        M_list.append(M0*1/2*(1-np.tanh((np.copy(Colloids.psi)-psi_c)*s)))
        psi_av.append(np.sum(np.copy(Colloids.psi))/(Dimensions[0]*Dimensions[1]))
        phi_av.append(np.sum(np.copy(Oil.phi))/(Dimensions[0]*Dimensions[1]))
    
    #Check progress at 10% intervals
    if ((n+1)*dt)%(t_sim/10)==0:
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
plt.ylabel('Av. Nanoparticle Content, $\psi_{av}$')

plt.plot(t_range,psi_av,label=r'$\psi_{av}$')
plt.plot(t_range,phi_av,label=r'$\phi_{av}$')

plt.legend()

#%% Visualise the 3D Morphology 

#z-Index
Index=0

Oil.Show(Index,MagentaBlack)

plt.figure()
plt.xlabel(r'Horizontal position, $\tilde{x}$')
plt.ylabel(r'Vertical position, $\tilde{y}$')
plt.imshow(Colloids.psi[Index],vmin=np.min(Colloids.psi[Index]),vmax=np.max(Colloids.psi[Index]),cmap=GreenBlack)
plt.colorbar()

plt.figure()
plt.xlabel(r'Vertical position, $\tilde{y}$')
plt.ylabel(r'Solvent Parameter, $\phi_s$')
plt.minorticks_on()

plt.plot(Alchohol_Profiles[len(Alchohol_Profiles)-1])
