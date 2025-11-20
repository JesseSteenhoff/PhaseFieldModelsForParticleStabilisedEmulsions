# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 18:58:39 2025

@author: J.M. Steenhoff
"""

#Import all the required modules 
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from scipy.stats import norm

#Preformat figures
plt.rcParams['figure.dpi'] = 300
plt.style.use(['science','no-latex'])

#Simulation time 
tsim=1000

#Simulation time range 
trange=np.arange(0,tsim,int(tsim/100))

#Number of time-varying distributions 
N=20

#Make a cumulative distribution over all time-varying distributions
Distribution_Cumulative_Colloids=[]
Distribution_Cumulative=[]
for i in range(len(trange)):
    Distribution_Cumulative_Colloids.append([])
    Distribution_Cumulative.append([])

#Import the time-varying distributions of the particle diameter 
#Path0_Colloids=<Insert path to 'Distributions_OffCritical_Colloids\psi0_0.30' >
#Path0=<Insert path to 'Distributions_OffCritical_NoColloids' >

for D in range(N):
    for i in range(len(trange)):
        
        Path_Colloids=Path0_Colloids+'\Dis'+str(D)+'\Distribution'+str(D)+'_'+str('{0:.1f}'.format(trange[i]))+'.txt'
        Distribution_Cumulative_Colloids[i]=np.concatenate((Distribution_Cumulative_Colloids[i], np.genfromtxt(Path_Colloids)))
        
        Path=Path0+'\Dis'+str(D)+'\Distribution'+str(D)+'_'+str('{0:.1f}'.format(trange[i]))+'.txt'
        Distribution_Cumulative[i]=np.concatenate((Distribution_Cumulative[i], np.genfromtxt(Path)))

#%% Make plot with stacked vertical histograms 

#Time (expressed in %)
T=[15,50,70,90]

#Figure size
W=8.5/2.54
H=9/2.54
FS=[W,H] 

#Fontsize 
fontsize=10
#Figure frame 
fig, axes = plt.subplots(nrows=len(T), ncols=1, sharex=True,figsize=FS)

plt.minorticks_on()
plt.xlabel(r'Droplet diameter, $\tilde{D}$',fontsize=fontsize)
fig.text(0.04, 0.5, 'Normalised Probability Density', va='center', rotation='vertical',fontsize=fontsize)

plt.subplots_adjust(hspace=0.04)

#Access the color-cycle 
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i in range(len(T)):

    #Fit Gaussian distributions
    av_Colloid,std_Colloid=norm.fit(Distribution_Cumulative_Colloids[T[i]])
    av,std=norm.fit(Distribution_Cumulative[T[i]])
    
    #Plot Gaussian distributions and histograms 
    D_range=np.arange(0,20,0.01)
    
    axes[i].plot(D_range, norm.pdf(D_range, av_Colloid, std_Colloid),color=color_cycle[i])
    axes[i].plot(D_range, norm.pdf(D_range, av, std),color=color_cycle[i])
    
    axes[i].hist(Distribution_Cumulative_Colloids[T[i]],bins=20,range=(np.min(Distribution_Cumulative_Colloids[T[i]]),np.max(Distribution_Cumulative_Colloids[T[i]])),alpha=0.55,rwidth=0.90,density='True',color=color_cycle[i])
    axes[i].hist(Distribution_Cumulative[T[i]],bins=20,range=(np.min(Distribution_Cumulative[T[i]]),np.max(Distribution_Cumulative[T[i]])),alpha=0.55,rwidth=0.90,density='True',color=color_cycle[i],histtype='step') 
    
    #Remove the labels of the ticks on the y-axis
    axes[i].set_yticklabels([])

#Make time-labels
Text_ys=[0.220,0.415,0.5975,0.7875][::-1]

for i in range(len(Text_ys)):

    fig.text(0.70, Text_ys[i], r'$\tilde{t}=$'+str(trange[T[i]]), va='center',color=color_cycle[i],fontsize=fontsize)
