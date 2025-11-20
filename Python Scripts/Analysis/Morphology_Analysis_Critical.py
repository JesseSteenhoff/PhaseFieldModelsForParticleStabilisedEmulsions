# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 19:35:03 2025

@author: J.M. Steenhoff
"""
#Import all the required modules 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import scipy.fft as FT
import scienceplots

from time import process_time
from scipy.optimize import fsolve
from matplotlib.colors import ListedColormap
from scipy.integrate import simpson 
from scipy.ndimage import gaussian_filter


#Preformat figures
plt.rcParams['figure.dpi'] = 300
plt.style.use(['science','no-latex'])

#%% Create a custom colourmap (magenta-black)

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

#%% The 'Liquid' class creates a phase-field object that evolves in accordance with the presented dynamic equations for jamming colloids

#The 'Liquid' class creates a PF object that evolves in accordance with a nondimensionalised Cahn-Hilliard equation 
class Liquid:
    
    #Binary interaction parameter
    chi=3                                 
    #Simulation timestep 
    dt=0.01
    #Stencil (grid) spacing 
    h=1
    
    #Initialisation method that creates a field  (Size[0]xSize[1]) of composition phi0, including some random thermal noise
    def __init__(self,Size,phi0):
        self.Size=Size
        self.phi0=phi0
        self.phi=self.phi0+np.random.randint(-10,10,(self.Size))*0.001
        
        #Solve the binodal equation to find the equilibrium compositions 
        self.phimax=fsolve(lambda x: np.log(x/(1-x))+self.chi*(1-2*x),0.99)
        self.phimin=1-self.phimax
    
    #Method that visualises current state of the phase-field (liquid)
    def Show(self,C):                           
        plt.figure()
        plt.xlabel(r'Horizontal position, $\tilde{x}$')
        plt.ylabel(r'Vertical position, $\tilde{y}$')
        plt.imshow(self.phi,norm=clr.Normalize(vmin=self.phimin,vmax=self.phimax),cmap=C)
        plt.colorbar()
        return None

    #Method that calculates the Laplacian of an input field 'F' via central finite-difference (5-point stencil)
    def Calc_Laplacian(self,F):               
        
        #Dimensions of the input field 
        Ny,Nx=F.shape
        
        #Create templates for the Laplacian contributions  
        Laplacian_x=np.zeros(F.shape)
        Laplacian_y=np.zeros(F.shape)
        
        #Apply periodic boundary conditions along the x-direction 
        Laplacian_x[:,1:Nx-1]=(-2*F[:,1:Nx-1]+F[:,2:Nx]+F[:,0:Nx-2])/(self.h**2)
        Laplacian_x[:,0]=(-2*F[:,0]+F[:,1]+F[:,Nx-1])/(self.h**2)                              
        Laplacian_x[:,Nx-1]=(-2*F[:,Nx-1]+F[:,Nx-2]+F[:,0])/(self.h**2)
        
        #Apply periodic boundary conditions along the y-direction
        Laplacian_y[1:Ny-1,:]=(-2*F[1:Ny-1,:]+F[2:Ny,:]+F[0:Ny-2,:])/(self.h**2)
        Laplacian_y[0,:]=(-2*F[0,:]+F[1,:]+F[Ny-1,:])/(self.h**2)                            
        Laplacian_y[Ny-1,:]=(-2*F[Ny-1,:]+F[Ny-2,:]+F[0,:])/(self.h**2)              
        
        Laplacian=Laplacian_x+Laplacian_y
    
        return Laplacian
    
    #Method that calculates the gradient vector for an input field 'F' via central finite-difference (5-point stencil)
    def Calc_Gradient(self,F):            
        
        #Dimensions of the input field 
        Ny,Nx=F.shape
        
        #Create templates for the gradient vector components  
        Gradientx=np.zeros(F.shape)                                                     
        Gradienty=np.zeros(F.shape)
        
        #Apply periodic boundary conditions along the x-direction 
        Gradientx[:,1:Nx-1]=(F[:,2:Nx]-F[:,0:Nx-2])/(2*self.h)
        Gradientx[:,0]=(F[:,1]-F[:,Nx-1])/(2*self.h)                                                        
        Gradientx[:,Nx-1]=(F[:,0]-F[:,Nx-2])/(2*self.h)
        
        #Apply periodic boundary conditions along the y-direction 
        Gradienty[1:Ny-1,:]=(F[2:Ny,:]-F[0:Ny-2,:])/(2*self.h)
        Gradienty[0,:]=(F[1,:]-F[Ny-1,:])/(2*self.h)                                  
        Gradienty[Ny-1,:]=(F[0,:]-F[Ny-2,:])/(2*self.h)        
        
        Gradient=(Gradientx,Gradienty)
        
        return Gradient
        
    #Method that evolves the order parameter field according to dynamic equations with jammed colloids 
    def Propagate(self,phi):
    
        #Calculates the functional derivative of the free energy (chemical potential) for the liquids        
        Mu=np.log(phi/(1-phi))+self.chi*(1-2*phi)-self.Calc_Laplacian(phi)
        
        #Update the order parameter field (Euler Forward)
        self.phi+=self.Calc_Laplacian(Mu)*self.dt
    
        return self.phi
    
    #Method that analyses the morphology (Find first moment of radially averaged power spectrum)
    def Analyse_Morphology(self):
        
        #Binarisation threshold 
        Bin=0.50
        #Binarise the morphology
        Morphology=np.copy(self.phi)
        Morphology[Morphology>=Bin]=1
        Morphology[Morphology<Bin]=0
        
        #Perform background substraction 
        Morphology=Morphology-gaussian_filter(Morphology, 100)
        
        #Extract dimensions
        D=Morphology.shape[0]
        
        #Perform 2D FFT to calculate 2D power spectrum 
        PS=np.abs(FT.fftshift(FT.fft2(Morphology)))**2
      
        #List with the indices of sampled frequencies
        Frequency_indices=np.arange(-int(D/2),int(D/2),1)
        #Convert frequency indices to actual spatial frequencies 
        Frequencies=Frequency_indices/D
        
        #Perform azimuthal averaging of the power spectrum 
        
        #Creates distance field centred around the centre of the power density spectrum.
        X,Y=np.meshgrid(range(D),range(D))
        R=np.sqrt(((X-int(D/2)))**2+(Y-int(D/2))**2)     
        
        #List with radial distances away from the centre (frequency indices!)
        Indices_list=np.arange(1,int(D/2),1)                       
        #Half-width of the radial area 
        dInd=0.5                                         
        
        #Storage list of the radial averages
        PS_Averaged=[]
                                              
        #Stores azimuthal average of the power density 
        for Ind in Indices_list:
            PS_Averaged.append(np.mean(PS[(R>=Ind-dInd) & (R<Ind+dInd)]))
        
        #List with radial distances away from the centre (spatial frequencies)
        Frequencies_radial=Indices_list/D
        
        #Calculate the first moment of the radially averaged power spectrum
        FM=simpson(PS_Averaged*Frequencies_radial,x=Frequencies_radial)/simpson(PS_Averaged,x=Frequencies_radial)
        
        #Calculate the characteristic length from the inverse of the first moment 
        Lmax=1/FM
        
        return Lmax
              
#%% Run phase-field simulation 

#Simulation dimensions 
Dimensions=(128,128)

#Initial composition (liquids()
phi0=0.50

#Simulated time
t_sim=200

#Storage list for the characteristic length 
Lcs=[]
    
#Initialise the liquid order parameter field 
Oil=Liquid(Dimensions,phi0)
      
#Storage lists for the morphologies 
Liquid_Morphology=[]

#Total number of simulation steps 
N=int(t_sim/Oil.dt)                  

#Start-point for determining computation time 
t0=process_time()  

for n in range(N+1):

    #Create local copies of the liquid field
    phi=np.copy(Oil.phi)

    #Propagate the fields in time via the dynamic equations
    Oil.Propagate(phi)
    
    #Export system morphology at 1% intervals
    if n%(N/100)==0:
        Liquid_Morphology.append(np.copy(Oil.phi))
        
        #Analyse morphology
        Lcs.append(Oil.Analyse_Morphology())
        
        print(str(100*n/N)+'% complete')
        
#End-point for determining computation time
t1=process_time()   
#Total computation time
Deltat=t1-t0         
print('Computation time: '+str(Deltat)+' s')

#%% Visualise the morphologies

#Simulation time expressed in %
#t_range=np.arange(0,110,10)
t_range=[100]

for i in t_range:
    plt.figure()
    plt.xlabel(r'Horizontal position, $\tilde{x}$')
    plt.ylabel(r'Vertical position, $\tilde{y}$')
    plt.title(r'Liquids, $\phi$')    
    plt.imshow(Liquid_Morphology[i],norm=clr.Normalize(vmin=Oil.phimin,vmax=Oil.phimax),cmap=MagentaBlack)
    plt.minorticks_on()
    plt.colorbar()
    
#%%Visualise coarsening profiles 

#Time-range
dt=t_sim/100
t_sims=np.arange(0,t_sim+dt,dt)

plt.figure()
plt.xlabel(r'Simulated time, $\tilde{t}$')
plt.ylabel(r'Average diameter, $\tilde{D}_{av}$')
plt.loglog(t_sims,Lcs,marker='o',linestyle='-',markersize=1.5)



    
    
    


