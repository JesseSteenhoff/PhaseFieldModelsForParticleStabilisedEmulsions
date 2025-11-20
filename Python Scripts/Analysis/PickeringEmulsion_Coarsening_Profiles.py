# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 09:49:35 2025

@author: J.M.Steenhoff
"""

#Import all the required modules 
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

#Preformat figures
plt.rcParams['figure.dpi'] = 300
plt.style.use(['science','no-latex'])

#Range over total number of profiles 
Profiles=20
Profile_range=range(Profiles)

#Insert paths to reference (no-colloids) and colloid-included data

#P0=<Insert path to 'Profiles_OffCritical_NoColloids'>
#P1=<Insert path to 'Profiles_OffCritical_Colloids'>

Path_Reference=P0+'\Profile'
Path_Data=P1

#Find number of datapoints in each profile 
Path=Path_Reference+str(Profile_range[0])+'.txt'
Points=len(np.genfromtxt(Path)[0])

#%%Prepare the reference data (average region number and diameter, averaged over all profiles)

N_ref=np.zeros(Points)
D_ref=np.zeros(Points)

#Import the data and average over all profiles 
for Profile in Profile_range:
    
    Path=Path_Reference+str(Profile)+'.txt'
    
    t_range,Ncs,Ds_av=np.genfromtxt(Path)
    
    for i in range(len(t_range)):
        D_ref[i]+=Ds_av[i]/Profiles
        N_ref[i]+=Ncs[i]/Profiles


#%%List with used initial values of the colloid field 
psi_list=[0.05,0.10,0.20,0.30,0.40,0.50]

#Prepare the colloid-included data 
N_list=[]
D_list=[]

for psi in psi_list:
    
    #Prepare the data (average region number and diameter, averaged over all profiles )
    N=np.zeros(Points)
    D=np.zeros(Points)

    #Import the data and average over all profiles 
    for Profile in Profile_range:
        
        Path=Path_Data+'\psi0_'+str('{0:.2f}'.format(psi))+'\Profile'+str(Profile)+'.txt'
        
        t_range,Ncs,Ds_av=np.genfromtxt(Path)
        
        for i in range(len(t_range)):
            D[i]+=Ds_av[i]/Profiles
            N[i]+=Ncs[i]/Profiles
    
    #Append the lists to storage 
    N_list.append(N)
    D_list.append(D)
    


#%%Visualise the evolution of the region number and the average diameter, averaged over all profiles 

plt.figure()
plt.minorticks_on()
plt.xlabel(r'Simulated time, $\tilde{t}$')
plt.ylabel('Droplet number, $N_d$')

#Trim data-set
C=2

#Plot the reference 
plt.plot(t_range[::C],N_ref[::C],marker='o',linestyle='-',markersize=1.5,label='0')
#plt.loglog(t_range[::C],N_ref[::C],marker='o',linestyle='--',markersize=1.5,label='0')
 

for i in range(len(N_list)):
    plt.plot(t_range[::C],N_list[i][::C],marker='o',linestyle='-',markersize=1.5,label=str(psi_list[i]))
    #plt.loglog(t_range[::C],N_list[i][::C],marker='o',linestyle='--',markersize=1.5,label=str(psi_list[i]))

plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.,title='$\psi_0$')

plt.figure()
plt.minorticks_on()
plt.xlabel(r'Simulated time, $\tilde{t}$')
plt.ylabel(r'Average diameter, $\tilde{D}_{av}$')

#Plot the reference 
plt.plot(t_range[::C],D_ref[::C],marker='o',linestyle='-',markersize=1.5,label='0')
#plt.loglog(t_range[::C],D_ref[::C],marker='o',linestyle='--',markersize=1.5,label='0')

for i in range(len(N_list)):
    plt.plot(t_range[::C],D_list[i][::C],marker='o',linestyle='-',markersize=1.5,label=str(psi_list[i]))
    #plt.loglog(t_range[::C],D_list[i][::C],marker='o',linestyle='--',markersize=1.5,label=str(psi_list[i]))

plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.,title='$\psi_0$')

#%% Plot the number and average diameter together

#Figure size
W=7.0/2.54 #W 7.5 for alt1
H=9/2.54
FS=[W,H] 

#Fontsize
Fontsize=10

#Figure frame 
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True,figsize=FS)
plt.minorticks_on()
plt.xlabel(r'Simulated time, $\tilde{t}$',fontsize=Fontsize)

axes[0].set_ylabel('Droplet number, $N_d$',fontsize=Fontsize)
axes[1].set_ylabel(r'Average diameter, $\tilde{D}_{av}$')

#Trim data-set
C=2

#Plot the references
axes[0].plot(t_range[::C],N_ref[::C],marker='o',linestyle='-',markersize=1.5,label='0')
axes[1].plot(t_range[::C],D_ref[::C],marker='o',linestyle='-',markersize=1.5,label='0')

for i in range(len(N_list)):
    axes[0].plot(t_range[::C],N_list[i][::C],marker='o',linestyle='-',markersize=1.5,label=str(psi_list[i]))
    axes[1].plot(t_range[::C],D_list[i][::C],marker='o',linestyle='-',markersize=1.5,label=str(psi_list[i]))

plt.subplots_adjust(hspace=0.02)

plt.legend(title='$\psi_0$',loc=(1.02,0.475))

