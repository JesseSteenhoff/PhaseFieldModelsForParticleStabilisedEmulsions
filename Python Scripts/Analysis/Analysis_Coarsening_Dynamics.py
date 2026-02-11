# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 18:28:46 2026

@author: J.M. Steenhoff
"""
#Import all the required modules 
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as FT
import scienceplots

from scipy.integrate import simpson 
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

from skimage.measure import label, regionprops
from skimage.segmentation import clear_border

#Preformat figures
plt.rcParams['figure.dpi'] = 300
plt.style.use(['science','no-latex'])

#Import path 
Path_Critical='<Path to Ostwald_Ripening_Critical Folder>'
Path_OffCritical='<Path to Ostwald_Ripening_OffCritical Folder>'


#%%Method that analyses the morphology (Find first moment of radially averaged power spectrum)
def Analyse_Morphology_Critical(Image):
    
    #Binarisation threshold 
    Bin=0.50
    #Binarise the morphology
    Morphology=np.copy(Image)
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

#Method that analyses the morphology to find the number of 'circles' present, along with their respective diameters
def Analyse_Morphology_OffCritical(Image):
    
    #Binarisation threshold 
    Bin=0.50
    #Binarise the morphology
    Morphology=np.copy(Image)
    Morphology[Morphology>=Bin]=1
    Morphology[Morphology<Bin]=0
    
    #Ignore any 'circles' that intersect with morphology boundaries
    Morphology=clear_border(Morphology)
    
    #Find connected regions in the morphology 
    Circles = label(Morphology, connectivity=2)
    
    #Get the properties for the different regions
    props = regionprops(Circles)
    
    #Extract the diameters, treating every found region as a circle 
    ds=[]
    
    for prop in props:
        #Area
        A=prop.area
        #Diameter
        d=np.sqrt(4*A/np.pi)
        ds.append(d)
    
    return np.average(ds)

#%% Import data and analyse 

#Number of simulations 
Ns=50

#Number of samples per simulation
Samples=100

#Time range of simulations 
t_sim=10000
t_range=np.arange(t_sim/Samples,t_sim+t_sim/Samples,t_sim/Samples)

#List with profile of characteristic length for each simulation 
Lcs_Critical=[]
Lcs_OffCritical=[]

for I in range(Ns):
    Lc_Critical=[]
    Lc_OffCritical=[]
    
    Morphologies_Critical=np.load(Path_Critical+'\Morphologies'+str(I)+'.npy')
    Morphologies_OffCritical=np.load(Path_OffCritical+'\Morphologies'+str(I)+'.npy')
    
    for M in range(Samples):
        Lc_Critical.append(Analyse_Morphology_Critical(Morphologies_Critical[M]))
        Lc_OffCritical.append(Analyse_Morphology_OffCritical(Morphologies_OffCritical[M]))
        
    Lcs_Critical.append(Lc_Critical)
    Lcs_OffCritical.append(Lc_OffCritical)
    
    print('Analysis of Simulation '+str(I+1)+' of '+str(Ns)+' completed')

#List with combined profiles of characteristic lengths 
Lcs_Critical_Total=list(zip(*Lcs_Critical))
Lcs_OffCritical_Total=list(zip(*Lcs_OffCritical))
#List with average profile of characteristic lengths 
Lcs_Critical_Average=[]
Lcs_OffCritical_Average=[]
#List with standard deviations of the characteristic lengths 
Lcs_Critical_Std=[]
Lcs_OffCritical_Std=[]

for i in range(Samples):
    Lcs_Critical_Average.append(np.average(Lcs_Critical_Total[i]))
    Lcs_OffCritical_Average.append(np.average(Lcs_OffCritical_Total[i]))
    Lcs_Critical_Std.append(np.std(Lcs_Critical_Total[i]))
    Lcs_OffCritical_Std.append(np.std(Lcs_OffCritical_Total[i]))
    
#%% Plot all profiles of the characteristic length 

plt.figure()
plt.minorticks_on()
plt.xlabel(r'Simulated time, $\tilde{t}$')
plt.ylabel(r'Characteristic length, $\tilde{L}_c$')

for i in range(Ns):
    if i==0:
        plt.loglog(t_range,Lcs_Critical[i],marker='o',linestyle='--',markersize=1,label='Critical',color='tab:blue')
        plt.loglog(t_range,Lcs_OffCritical[i],marker='o',linestyle='--',markersize=1,label='Off-critical',color='tab:green')
    else:
        plt.loglog(t_range,Lcs_Critical[i],marker='o',linestyle='--',markersize=1,color='tab:blue',alpha=0.50)
        plt.loglog(t_range,Lcs_OffCritical[i],marker='o',linestyle='--',markersize=1,color='tab:green',alpha=0.50)

plt.legend()

#%%Plot the averaged profile of the characteristic length 

plt.figure()
plt.minorticks_on()
plt.xlabel(r'Simulated time, $\tilde{t}$')
plt.ylabel(r'Characteristic length, $\tilde{L}_c$')


plt.loglog(t_range,Lcs_Critical_Average,marker='o',linestyle='--',markersize=1,label='Critical')
plt.loglog(t_range,Lcs_OffCritical_Average,marker='o',linestyle='--',markersize=1,label='Off-critical')

plt.legend()

#%% Set a minimum time for analysis and visualise again 

#Time cut-off
t_min_Critical=100
t_min_OffCritical=100

#Create masks
mask_Critical=t_range>=t_min_Critical
mask_OffCritical=t_range>=t_min_OffCritical 

#Create masked arrays 
t_cut_critical=t_range[mask_Critical]
t_cut_offcritical=t_range[mask_OffCritical]

Lcs_Critical_cut=[]
Lcs_OffCritical_cut=[]

for i in range(Ns):
    Lcs_Critical_cut.append(np.array(Lcs_Critical[i])[mask_Critical])
    Lcs_OffCritical_cut.append(np.array(Lcs_OffCritical[i])[mask_OffCritical])

Lcs_Critical_Average_cut=np.array(Lcs_Critical_Average)[mask_Critical]
Lcs_OffCritical_Average_cut=np.array(Lcs_OffCritical_Average)[mask_OffCritical]

plt.figure()
plt.minorticks_on()
plt.xlabel(r'Simulated time, $\tilde{t}$')
plt.ylabel(r'Characteristic length, $\tilde{L}_c$')

for i in range(Ns):
    if i==0:
        plt.loglog(t_cut_critical,Lcs_Critical_cut[i],marker='o',linestyle='--',markersize=1,label='Critical',color='tab:blue')
        plt.loglog(t_cut_offcritical,Lcs_OffCritical_cut[i],marker='o',linestyle='--',markersize=1,label='Off-critical',color='tab:green')
    else:
        plt.loglog(t_cut_critical,Lcs_Critical_cut[i],marker='o',linestyle='--',markersize=1,color='tab:blue',alpha=0.50)
        plt.loglog(t_cut_offcritical,Lcs_OffCritical_cut[i],marker='o',linestyle='--',markersize=1,color='tab:green',alpha=0.50)

plt.legend()

#%% Extract coarsening exponent of the averaged profiles with an exponential model

#Exponential Model 
def Model(x,b,a):
    return a*x**(b)

#Fitting range 
t_fit_Critical=np.arange(t_min_Critical,np.max(t_cut_critical),0.01)
t_fit_OffCritical=np.arange(t_min_OffCritical,np.max(t_cut_offcritical),0.01)

Beta_c,Prefactor_c=curve_fit(Model,t_cut_critical, Lcs_Critical_Average_cut)[0]
Beta_oc,Prefactor_oc=curve_fit(Model,t_cut_offcritical, Lcs_OffCritical_Average_cut)[0]

plt.figure()
plt.minorticks_on()
plt.xlabel(r'Simulated time, $\tilde{t}$')
plt.ylabel(r'Characteristic length, $\tilde{L}_c$')

plt.loglog(t_cut_critical,Lcs_Critical_Average_cut,marker='o',linestyle='',markersize=1)
plt.loglog(t_cut_critical,Model(t_cut_critical,Beta_c,Prefactor_c),color=plt.gca().lines[-1].get_color(),linestyle='--',label='Critical')

plt.loglog(t_cut_offcritical,Lcs_OffCritical_Average_cut,marker='o',linestyle='',markersize=1)
plt.loglog(t_cut_offcritical,Model(t_cut_offcritical,Beta_oc,Prefactor_oc),color=plt.gca().lines[-1].get_color(),linestyle='--',label='Off-critical')

plt.legend()

print(Beta_c,Beta_oc)


#%%Fit exponential model to each individual profile and extract coarsening exponents 

#List with coarsening exponents 
Beta_Critical=[]
Beta_OffCritical=[]

#List with prefactors  
Prefactor_Critical=[]
Prefactor_OffCritical=[]

for i in range(Ns):
    Bc,Pc=curve_fit(Model, t_cut_critical, Lcs_Critical_cut[i])[0]
    Boc,Poc=curve_fit(Model, t_cut_offcritical, Lcs_OffCritical_cut[i] )[0]
    
    Beta_Critical.append(Bc)
    Beta_OffCritical.append(Boc)
    
    Prefactor_Critical.append(Pc)
    Prefactor_OffCritical.append(Poc)


#Calculate average coarsening exponents and their standard deviation
Beta_Critical_Average=np.average(Beta_Critical)
Beta_Critical_Std=np.std(Beta_Critical)

Beta_OffCritical_Average=np.average(Beta_OffCritical)
Beta_OffCritical_Std=np.std(Beta_OffCritical)

Prefactor_Critical_Average=np.average(Prefactor_Critical)
Prefactor_OffCritical_Average=np.average(Prefactor_OffCritical)

plt.figure()
plt.minorticks_on()
plt.xlabel('Sample Number')
plt.ylabel(r'Coarsening Exponent, $\beta$')

plt.hlines(1/3,0,len(Beta_Critical),color='tab:red',linestyle='--')
plt.hlines(Beta_Critical_Average,0,len(Beta_Critical),color='tab:blue',linestyle='--')

plt.plot(Beta_Critical,marker='o',markersize=3,linestyle='',label='Critical')
plt.plot(Beta_OffCritical,marker='o',markersize=3,linestyle='',label='Off-critical')


#%% Visualise the coarsening exponent versus initial liquid composition

#Initial Liquid Composition
phi0s=(0.25,0.50)
#Coarsening exponents 
Betas=(Beta_OffCritical_Average,Beta_Critical_Average)
Stds=(Beta_OffCritical_Std,Beta_Critical_Std)

plt.figure()
plt.minorticks_on()
plt.xlabel(r'Liquid composition, $\phi_0$')
plt.ylabel(r'Coarsening exponent, $\beta$')

plt.errorbar(phi0s, Betas , yerr=Stds,linestyle='',capsize=2,color='tab:blue',marker='o',markersize=5)
plt.plot([0.50],[1/3],linestyle='',marker='x',markersize=6,color='black')

#plt.savefig(Path_Exp+'\CoarseningExponents_Comparison_300dpi.png',dpi=300)

#%% Plot average profile of the characteristic length with fitted coarsening exponent (+prefactor fitted from average)

plt.figure()
plt.minorticks_on()
plt.xlabel(r'Simulated time, $\tilde{t}$')
plt.ylabel(r'Characteristic length, $\tilde{L}_c$')

plt.loglog(t_cut_critical,Lcs_Critical_Average_cut,marker='o',linestyle='',markersize=2)
plt.loglog(t_cut_critical,Model(t_cut_critical,Beta_Critical_Average,Prefactor_c),color=plt.gca().lines[-1].get_color(),linestyle='--',label='Critical')

plt.loglog(t_cut_offcritical,Lcs_OffCritical_Average_cut,marker='o',linestyle='',markersize=2)
plt.loglog(t_cut_offcritical,Model(t_cut_offcritical,Beta_OffCritical_Average,Prefactor_oc),color=plt.gca().lines[-1].get_color(),linestyle='--',label='Critical')


print(Beta_Critical_Average,Beta_Critical_Std)
print(Beta_OffCritical_Average,Beta_OffCritical_Std)

#%% Previous plot with some additional text 

plt.figure()
plt.minorticks_on()
plt.xlabel(r'Simulated time, $\tilde{t}$')
plt.ylabel(r'Characteristic length, $\tilde{L}_c$')

plt.loglog(t_cut_critical,Lcs_Critical_Average_cut,marker='o',linestyle='',markersize=2)
plt.loglog(t_cut_critical,Model(t_cut_critical,Beta_Critical_Average,Prefactor_c),color=plt.gca().lines[-1].get_color(),linestyle='--',label='Critical')
plt.text(650,31,r'$\beta='+str(np.round(Beta_Critical_Average,2))+'\pm '+str(np.round(Beta_Critical_Std,2))+'$',color=plt.gca().lines[-1].get_color(),rotation=26)

plt.loglog(t_cut_offcritical,Lcs_OffCritical_Average_cut,marker='o',linestyle='',markersize=2)
plt.loglog(t_cut_offcritical,Model(t_cut_offcritical,Beta_OffCritical_Average,Prefactor_oc),color=plt.gca().lines[-1].get_color(),linestyle='--',label='Off-critical')
plt.text(650,13,r'$\beta='+str(np.round(Beta_OffCritical_Average,2))+'\pm '+str(np.round(Beta_OffCritical_Std,2))+'$',color=plt.gca().lines[-1].get_color(),rotation=20)

plt.legend()



