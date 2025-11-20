# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 18:09:13 2025

@author: J.M. Steenhoff
"""

#Import all the required modules 
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.optimize import curve_fit

#Preformat figures
plt.rcParams['figure.dpi'] = 300
plt.style.use(['science','no-latex'])

#Number of profiles
N=60

#Range over total number of profiles 
Profile_range=range(N)

#Simulation time 
t_sim=10000

#Specificy base paths of data files
BasePath_Spin=<Insert Path to 'Profiles_OstwaldRipening_Critical' >
BasePath_Drop=<Insert Path to 'Profiles_OstwaldRipening_Offcritical' >


#Find number of datapoints in each profile 
Path_Spin=BasePath_Spin+r'\Profile'+str(Profile_range[0])+'.txt'
Path_Drop=BasePath_Drop+r'\Profile'+str(Profile_range[0])+'.txt'
Points=len(np.genfromtxt(Path_Spin)[0])

#List of characteristic length profiles 
Lcs_Spin=[]
Lcs_Drop=[]

#List with average characteristic length over all profiles 
Lcs_Spin_Average=np.zeros(Points)
Lcs_Drop_Average=np.zeros(Points)


#Import the data and average over all profiles 
for Profile in Profile_range:
    
    Path_Spin=BasePath_Spin+r'\Profile'+str(Profile)+'.txt'
    Path_Drop=BasePath_Drop+r'\Profile'+str(Profile)+'.txt'
    
    t_range,Lc=np.genfromtxt(Path_Spin)
    t_range,Ncs,Ds_av=np.genfromtxt(Path_Drop)
    
    Lcs_Spin.append(Lc)
    Lcs_Drop.append(Ds_av)
            
    #Average over profiles
    for i in range(len(t_range)):
        Lcs_Spin_Average[i]+=Lc[i]/N
        Lcs_Drop_Average[i]+=Ds_av[i]/N

#Calculate the standard devation for each point from all profiles 
Stds_Spin=[]
Stds_Drop=[]

for i in range(Points):
    Lc_Point_Spin=[]
    Lc_Point_Drop=[]
    for j in range(N):
        Lc_Point_Spin.append(Lcs_Spin[j][i])
        Lc_Point_Drop.append(Lcs_Drop[j][i])  
    Stds_Spin.append(np.std(Lc_Point_Spin))
    Stds_Drop.append(np.std(Lc_Point_Drop))
    
#%% Plot all the profiles of the characteristic lengths 

plt.figure()
plt.minorticks_on()
plt.xlabel(r'Simulated time, $\tilde{t}$')
plt.ylabel(r'Characteristic length, $\tilde{L}_c$')

for i in range(N):
    if i==0:
        plt.loglog(t_range,Lcs_Spin[i],marker='o',linestyle='',markersize=1,color='tab:blue',label='Critical')
        plt.loglog(t_range,Lcs_Drop[i],marker='o',linestyle='',markersize=1,color='tab:green',label='Off-critical')
    else:
        plt.loglog(t_range,Lcs_Spin[i],marker='o',linestyle='',markersize=1,color='tab:blue')
        plt.loglog(t_range,Lcs_Drop[i],marker='o',linestyle='',markersize=1,color='tab:green')

plt.legend(frameon='')

#%% Obtain rate exponents by fitting all profiles

#Time cut-off
t_min_Spin=50
t_min_Drop=100

#Create masks
mask_Spin=t_range>=t_min_Spin
mask_Drop=t_range>=t_min_Drop

#Exponential model
def Model(x,b,a):
    return a*x**(b)

#List of exponents 
alphas_Spin=[]
alphas_Drop=[]

#List of prefactors 
PFs_Spin=[]
PFs_Drop=[]

for i in range(N):
    alpha_Spin,PF_spin=curve_fit(Model, t_range[mask_Spin], Lcs_Spin[i][mask_Spin],p0=(1/3,2.8))[0]
    alpha_Drop,PF_drop=curve_fit(Model, t_range[mask_Drop], Lcs_Drop[i][mask_Drop])[0]
    
    alphas_Spin.append(alpha_Spin)
    alphas_Drop.append(alpha_Drop)
    
    PFs_Spin.append(PF_spin)
    PFs_Drop.append(PF_drop)

#Averages
alpha_Spin_Average=np.average(alphas_Spin)
alpha_Drop_Average=np.average(alphas_Drop)

PFs_Spin_Average=np.average(PFs_Spin)
PFs_Drop_Average=np.average(PFs_Drop)

#Standard deviations 
Std_Spin=np.std(alphas_Spin)
Std_Drop=np.std(alphas_Drop)

#%% Visualise the exponent rate versus initial liquid composition

phi0s=(0.25,0.50)
Alphas=(alpha_Drop_Average,alpha_Spin_Average)
Stds=(Std_Drop,Std_Spin)

plt.figure()
plt.minorticks_on()
plt.xlabel(r'Liquid composition, $\phi_0$')
plt.ylabel(r'Rate exponent, $\beta$')

for i in range(N):
    plt.plot([0.25],[alphas_Drop[i]],marker='o',markersize=3,color='tab:blue')
    plt.plot([0.50],[alphas_Spin[i]],marker='o',markersize=3,color='tab:blue')

plt.plot([0.50],[1/3],linestyle='',marker='x',markersize=4,color='black')
#plt.plot(phi0s,Alphas,linestyle='',marker='o',markersize=4,color='black')

plt.figure()
plt.minorticks_on()
plt.xlabel(r'Liquid composition, $\phi_0$')
plt.ylabel(r'Coarsening exponent, $\beta$')

plt.errorbar(phi0s, Alphas , yerr=Stds,linestyle='',capsize=2,color='tab:blue',marker='o',markersize=5)
plt.plot([0.50],[1/3],linestyle='',marker='x',markersize=6,color='black')


#%% Plot all averaged profiles of the characteristic length 

plt.figure()
plt.minorticks_on()
plt.xlabel(r'Simulated time, $\tilde{t}$')
plt.ylabel(r'Characteristic length, $\tilde{L}_c$')

plt.loglog(t_range,Lcs_Spin_Average,marker='o',linestyle='--',markersize=2,label='Critical')
plt.loglog(t_range,Lcs_Drop_Average,marker='o',linestyle='--',markersize=2,label='Off-critical')

plt.legend(frameon='')

plt.figure()
plt.minorticks_on()
plt.xlabel(r'Simulated time, $\tilde{t}$')
plt.ylabel(r'Characteristic length, $\tilde{L}_c$')

plt.plot(t_range,Lcs_Spin_Average,marker='o',linestyle='--',markersize=1,label='Critical')
plt.errorbar(t_range,Lcs_Spin_Average,yerr=Stds_Spin,linestyle='',capsize=1,color=plt.gca().lines[-1].get_color())

plt.plot(t_range,Lcs_Drop_Average,marker='o',linestyle='--',markersize=1,label='Off-critical')
plt.errorbar(t_range,Lcs_Drop_Average,yerr=Stds_Drop,linestyle='',capsize=1,color=plt.gca().lines[-1].get_color())

plt.legend(frameon='')

#%% Plot the averaged profiles with the averaged fitting parameters (exponent, prefactor)

plt.figure()
plt.minorticks_on()
plt.xlabel(r'Simulated time, $\tilde{t}$')
plt.ylabel(r'Characteristic length, $\tilde{L}_c$')

#Fit parameters of the averaged curves
Alpha_Spin,PF_Spin=curve_fit(Model,t_range[mask_Spin],Lcs_Spin_Average[mask_Spin],p0=(1/3,2.8))[0]
Alpha_Drop,PF_Drop=curve_fit(Model,t_range[mask_Drop],Lcs_Drop_Average[mask_Drop])[0]

plt.loglog(t_range,Lcs_Spin_Average,marker='o',linestyle='',markersize=2)
#Plot fits with an exponent averaged from all profiles and a prefactor from the averaged profile
plt.loglog(t_range[mask_Spin],Model(t_range,alpha_Spin_Average,PF_Spin)[mask_Spin],color=plt.gca().lines[-1].get_color(),linestyle='--',label='Critical')

plt.loglog(t_range,Lcs_Drop_Average,marker='o',linestyle='',markersize=2)
#Plot fits with an exponent averaged from all profiles and a prefactor from the averaged profile
plt.loglog(t_range[mask_Drop],Model(t_range,alpha_Drop_Average,PF_Drop)[mask_Drop],color=plt.gca().lines[-1].get_color(),linestyle='--',label='Off-critical')

plt.legend(frameon='')

#%%Plot some additinal texts with the coarsening exponents 

plt.figure()
plt.minorticks_on()
plt.xlabel(r'Simulated time, $\tilde{t}$')
plt.ylabel(r'Characteristic length, $\tilde{L}_c$')

plt.loglog(t_range,Lcs_Spin_Average,marker='o',linestyle='',markersize=2)
#Plot fits with an exponent averaged from all profiles and a prefactor from the averaged profile
plt.loglog(t_range[mask_Spin],Model(t_range,alpha_Spin_Average,PF_Spin)[mask_Spin],color=plt.gca().lines[-1].get_color(),linestyle='--',label='Critical')
plt.text(1000,8,r'$\beta=0.35\pm 0.03$',color=plt.gca().lines[-1].get_color())

plt.loglog(t_range,Lcs_Drop_Average,marker='o',linestyle='',markersize=2)
#Plot fits with an exponent averaged from all profiles and a prefactor from the averaged profile
plt.loglog(t_range[mask_Drop],Model(t_range,alpha_Drop_Average,PF_Drop)[mask_Drop],color=plt.gca().lines[-1].get_color(),linestyle='--',label='Off-critical')
plt.text(1000,5,r'$\beta=0.26\pm 0.03$',color=plt.gca().lines[-1].get_color())

plt.legend(frameon='')
   