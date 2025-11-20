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

from time import process_time
from scipy.optimize import fsolve
from matplotlib.colors import ListedColormap
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage.color import label2rgb


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
    def __init__(self,Size,phi0,alpha):
        self.Size=Size
        self.phi0=phi0
        self.phi=self.phi0+np.random.randint(-10,10,(self.Size))*0.001
        
        # Attachment parameter
        self.alpha = alpha
        
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
    
    #Method that modulates the liquid mobility based on the presence of colloids (jamming)
    def Calc_Mobility(self,psi,psic,n):
        
        M=1/2*(1-np.tanh((psi-psic)*n))
    
        return M
    
    #Method that evolves the order parameter field according to dynamic equations with jammed colloids 
    def Propagate(self,phi,psi,psic,n):
    
        #Calculates the functional derivative of the free energy (chemical potential) for the liquids        
        Mu=np.log(phi/(1-phi))+self.chi*(1-2*phi)-self.Calc_Laplacian(phi)
        
        #Update the order parameter field (Euler Forward)
        self.phi+=(self.Calc_Mobility(psi, psic, n)*self.Calc_Laplacian(Mu))*self.dt
    
        return self.phi
    
    #Method that analyses the morphology to find the number of 'circles' present, along with their respective diameters
    def Analyse_Morphology(self):
        
        #Binarisation threshold 
        Bin=0.50
        #Binarise the morphology
        Morphology=np.copy(self.phi)
        Morphology[Morphology>=Bin]=1
        Morphology[Morphology<Bin]=0
        
        #Ignore any 'circles' that intersect with morphology boundaries
        Morphology=clear_border(Morphology)
        
        #Find connected regions in the morphology 
        Circles = label(Morphology, connectivity=2)
        
        #Get the properties for the different regions
        props = regionprops(Circles)
        
        #Count the different regions 
        Nc=len(props)
        
        #Extract the diameters, treating every found region as a circle 
        ds=[]
        
        for prop in props:
            #Area
            A=prop.area
            #Diameter
            d=np.sqrt(4*A/np.pi)
            ds.append(d)
        
        #Make an rgb overlay of the found regions 
        Overlay=label2rgb(Circles,Oil.phi,colors=[[1,0,1]],bg_color=None,bg_label=0,kind='overlay')
        
        return (Nc,ds,Overlay)

#The Colloid class describes the behaviour of surface-active, jamming nanoparticles in the system.
class Colloid(Liquid):
    
    #Relative mobility of the colloids
    Mc=0.01
    
    #Initisalisation method for the Colloid field (child class of 'Liquid')
    def __init__(self,Size,psi0,alpha):
        
        #Initialise parent class (Field)
        Liquid.__init__(self,Size,phi0,alpha)
        
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
    
    #Method that modulates the colloid mobility (jamming)
    def Calc_Mobility_C(self,psi,psic,n):
        
        MC=self.Mc*1/2*(1-np.tanh((psi-psic)*n))

        return MC
    
    #Method that propagates the colloid order parameter field in time according to the dynamic equations with jamming colloids
    def Propagate(self,psi,phi,psic,n):
        
        #Calculate the gradient of the liquid field 
        Gradx,Grady=self.Calc_Gradient(phi)
        
        #Calculates the functional derivative of the free energy (chemical potential) of the nanoparticles
        Mu=np.log(psi)-self.alpha/2*(Gradx**2+Grady**2)
        
        #Calculate the mobility of the colloids 
        MC=self.Calc_Mobility_C(psi, psic, n)
        
        #Update order parameter field (Euler Forward)
        self.psi+=(MC*self.Calc_Laplacian(Mu))*self.dt
    
        return self.psi

              
#%% Run phase-field simulation 

#Simulation dimensions 
Dimensions=(128,128)

#Initial composition (liquids, colloids)
phi0=0.25
psi0=0.20

#Attachment parameter 
Alpha=60

#Threshold value for colloid jamming
psi_c=0.60
#''Sharpness'' of colloid mobility function
s=50

#Simulated time
t_sim=200

#Storage list for the number of found regions 
Ncs=[]
#Storage list for the diameters of the found regions 
Ds=[]
#Storage list for the average diameters of the found regions 
Ds_av=[]
#Storage list for the overlays of found regions
Overlays=[]
    
#Initialise the liquid order parameter field 
Oil=Liquid(Dimensions,phi0,Alpha)

#Initialise the colloid field 
Colloids=Colloid(Dimensions,psi0,Alpha)         

#Storage lists for the morphologies 
Liquid_Morphology=[]
Colloid_Morphology=[]

#Storage list for the mobility
M_list=[]

#Total number of simulation steps 
N=int(t_sim/Oil.dt)                  

#Start-point for determining computation time 
t0=process_time()  

for n in range(N+1):

    #Create local copies of the liquid and colloid field 
    phi,psi=np.copy(Oil.phi),np.copy(Colloids.psi)

    #Propagate the fields in time via the dynamic equations for jamming particles
    Oil.Propagate(phi,psi,psi_c,s)
    Colloids.Propagate(psi,phi,psi_c,s)
    
    #Export system morphology at 1% intervals
    if n%(N/100)==0:
        Liquid_Morphology.append(np.copy(Oil.phi))
        Colloid_Morphology.append(np.copy(Colloids.psi))
        M_list.append(Oil.Calc_Mobility(psi, psi_c, s))
        
        #Analyse morphology
        nc,ds,O=Oil.Analyse_Morphology()
        #Export the number of found regions, their diameters and the average diameter
        Ncs.append(nc)
        Ds.append(ds)
        Overlays.append(O)
        if len(ds)==0:
            Ds_av.append(0)
        else:
            Ds_av.append(np.average(ds))
        
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
    
    plt.figure()
    plt.xlabel(r'Horizontal position, $\tilde{x}$')
    plt.ylabel(r'Vertical position, $\tilde{y}$')
    plt.title(r'Liquids, $\phi$')    
    plt.imshow(Overlays[i])
    plt.minorticks_on()
    
    plt.figure()
    plt.xlabel(r'Horizontal position, $\tilde{x}$')
    plt.ylabel(r'Vertical position, $\tilde{y}$')
    plt.title(r'Nanoparticles, $\psi$')
    #plt.tick_params(colors='white', which='both')
    plt.imshow(Colloid_Morphology[i],vmin=0,vmax=0.6,cmap=GreenBlack)
    plt.minorticks_on()
    plt.colorbar()

#%%Visualise coarsening profiles 

#Time-range
dt=t_sim/100
t_sims=np.arange(0,t_sim+dt,dt)

plt.figure()
plt.xlabel(r'Simulated time, $\tilde{t}$')
plt.ylabel(r'Average diameter, $\tilde{D}_{av}$')
plt.plot(t_sims,Ds_av,marker='o',linestyle='-',markersize=1.5)

plt.figure()
plt.xlabel(r'Simulated time, $\tilde{t}$')
plt.ylabel(r'Droplet number, $N_d$')
plt.plot(t_sims,Ncs,marker='o',linestyle='-',markersize=1.5)


    
    
    


