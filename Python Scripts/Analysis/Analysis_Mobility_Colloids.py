# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 19:10:38 2025

@author: J.M. Steenhoff
"""
#Import all the required modules 
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as FT
import scienceplots

from scipy.integrate import simpson 
from scipy.ndimage import gaussian_filter

#Preformat figures
plt.rcParams['figure.dpi'] = 300
plt.style.use(['science','no-latex'])

#Import path 
Path='<Path to Profiles_Mobility_Variation_Colloids Folder'

#Method that analyses the morphology (Find first moment of radially averaged power spectrum)
def Analyse_Morphology(Image):
    
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

#%% Import data and analyse 

#Number of simulations 
Ns=10

#Colloid mobilities 
Mc=[1/100,1/500,1/1000,1/2000]

#List with characteristic lengths 
Lcs=[[] for _ in range(len(Mc))]

for M in range(len(Mc)):
    for I in range(Ns):
        Lc=[]
        Morphologies=np.load(Path+'\Mc='+str(Mc[M])+'\Morphologies'+str(I)+'.npy')
        for Morphology in Morphologies:
            Lc.append(Analyse_Morphology(Morphology))
        Lcs[M].append(Lc)

#Calculate average and standard deviation 
Lcs_av=[[] for _ in range(len(Mc))]
Lcs_std=[[] for _ in range(len(Mc))]

for i in range(len(Mc)):
    Total=list(zip(*Lcs[i]))
    for j in range(len(Total)):
        Lcs_av[i].append(np.average(Total[j]))
        Lcs_std[i].append(np.std(Total[j]))

#Import reference data 
Lcs_ref=[[] for _ in range(len(Lcs_av[0]))]
Lcs_ref_av=[[] for _ in range(len(Lcs_av[0]))]
Lcs_ref_std=[[] for _ in range(len(Lcs_std[0]))]

for I in range(Ns):
    Morphologies=np.load(Path+'\Reference\Morphologies'+str(I)+'.npy')
    for i in range(Morphologies.shape[0]):
        Lcs_ref[i].append(Analyse_Morphology(Morphologies[i]))
for i in range(len(Lcs_ref)):
    Lcs_ref_av[i]=np.average(Lcs_ref[i])
    Lcs_ref_std[i]=np.std(Lcs_ref[i])

#%% Visualise evolution of the characteristic length for different colloid mobilities 

#Simulated time 
tsim=500
#Time step
dt=tsim/(len(Lcs_av[0])-1)
#Time range
t_range=np.arange(0,tsim+dt,dt)

plt.figure()
plt.minorticks_on()
plt.xlabel(r'Simulated time, $\tilde{t}$')
plt.ylabel(r'Characteristic length, $\tilde{L}_c$')

plt.errorbar(t_range, Lcs_ref_av,Lcs_ref_std,marker='o',markersize=1,linestyle=':',label='0',capsize=1)

for i in range(len(Lcs_av))[::-1]:
    
    plt.errorbar(t_range, Lcs_av[i],Lcs_std[i],marker='o',markersize=1,linestyle=':',label=str(Mc[i]),capsize=1)



plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0.,title=r'$\tilde{M}_{\psi}^0$')



