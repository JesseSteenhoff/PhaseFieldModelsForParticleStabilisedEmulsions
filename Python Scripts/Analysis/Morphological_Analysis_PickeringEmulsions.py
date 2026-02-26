# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 16:41:14 2026

@author: J.M. Steenhoff
"""

#Import all the required modules 
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from scipy.stats import norm
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border


#Preformat figures
plt.rcParams['figure.dpi'] = 300
plt.style.use(['science','no-latex'])

#Import Path for data 
Path_Folder='<Path to Profiles_Pickering_Emulsions Folder>' 
Path_Im=Path_Folder+'\psi0='

#%% Analysis function for the droplet morphology 

#Function that analyses the morphology to find the number of 'circles' present, along with their respective diameters
def Analyse_Morphology(Image):
    
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
        
    return (Nc,np.array(ds))

#%% Import and analyse the data 

#Initial nanoparticle concentrations 
psi0_range=[0,0.05,0.10,0.20,0.30,0.40,0.50]

#Number of simulations per nanoparticle concentration
N=25

#Storage list for the circle number and diameter distributions for each nanoparticle concentration, for all simulations
NCs=[]
DSs=[]
DSs_av=[]

for psi0 in psi0_range:
    
    #Local list for the circle number and diameter distributions for each nanoparticle concentration
    NC_list_T=[]
    DS_list_T=[]
    DS_av_list_T=[]
    
    for n in range(N):
        
        #Local list for the circle number and diameter distributions for each simulation
        NC_list=[]
        DS_list=[]
        DS_av_list=[]
        
        #Import time-set of morphologies
        if psi0==0:
            Morphologies1=np.load(Path_Im+str(psi0)+'\Morphologies_Liquid_'+str(n)+'_Part1.npy')
            Morphologies2=np.load(Path_Im+str(psi0)+'\Morphologies_Liquid_'+str(n)+'_Part2.npy')
            Morphologies3=np.load(Path_Im+str(psi0)+'\Morphologies_Liquid_'+str(n)+'_Part3.npy')
            
            Morphologies=np.concatenate((Morphologies1,Morphologies2,Morphologies3),axis=0)
            
        else:            
            Morphologies1=np.load(Path_Im+str('{:.2f}'.format(psi0))+'\Morphologies_Liquid_'+str(n)+'_Part1.npy')
            Morphologies2=np.load(Path_Im+str('{:.2f}'.format(psi0))+'\Morphologies_Liquid_'+str(n)+'_Part2.npy')
            Morphologies3=np.load(Path_Im+str('{:.2f}'.format(psi0))+'\Morphologies_Liquid_'+str(n)+'_Part3.npy')
            
            Morphologies=np.concatenate((Morphologies1,Morphologies2,Morphologies3),axis=0)
            
        #Analyse each morphology over time 
        for i in range(Morphologies.shape[0]):
            NC,DS=Analyse_Morphology(Morphologies[i])
            NC_list.append(NC)
            DS_list.append(DS)
            if len(DS)==0:
                DS_av_list.append(0)
            else:
                DS_av_list.append(np.average(DS))
        
        NC_list_T.append(NC_list)
        DS_list_T.append(DS_list)
        DS_av_list_T.append(DS_av_list)
    
    NCs.append(NC_list_T)
    DSs.append(DS_list_T)
    DSs_av.append(DS_av_list_T)
    
    print('psi0='+str('{:.2f}'.format(psi0))+' completed!')

    
#Calculate profiles (both averaged and cumulative) of the circle number and diameter distributions  
Profiles_NC_Average=[]
Profiles_NC_Std=[]

Profiles_DS_Average=[]
Profiles_DS_Std=[]
Profiles_DS_Cumulative=[]

for i in range(len(psi0_range)):
    Profile_NC_Cumulative=[[] for _ in range(len(NCs[0][0]))]
    Profile_DS_Cumulative=[[] for _ in range(len(DSs[0][0]))]
    Profile_DS_Cumulative_Av=[[] for _ in range(len(DSs[0][0]))]
    for j in range(N):
        for k in range(len(NCs[0][0])):
            Profile_NC_Cumulative[k].append(NCs[i][j][k])
            Profile_DS_Cumulative_Av[k].append(DSs_av[i][j][k])
            Profile_DS_Cumulative[k].append(DSs[i][j][k])

    Profile_NC_Average=[]
    Profile_NC_Std=[]
    
    Profile_DS_Average=[]
    Profile_DS_Std=[]
    
    for l in range(len(Profile_NC_Cumulative)):
        Profile_NC_Average.append(np.average(Profile_NC_Cumulative[l]))
        Profile_NC_Std.append(np.std(Profile_NC_Cumulative[l]))
        
        Profile_DS_Average.append(np.average(Profile_DS_Cumulative_Av[l]))
        Profile_DS_Std.append(np.std(Profile_DS_Cumulative_Av[l]))
    
    Profiles_NC_Average.append(Profile_NC_Average)
    Profiles_NC_Std.append(Profile_NC_Std)
    
    Profiles_DS_Average.append(Profile_DS_Average)
    Profiles_DS_Std.append(Profile_DS_Std)
    
    Profiles_DS_Cumulative.append(Profile_DS_Cumulative)


#%% Visualise the found results 

#Time-range
t_sim=1000
t_range=np.arange(t_sim/100,t_sim+t_sim/100,t_sim/100)

plt.figure()
plt.minorticks_on()
plt.xlabel(r'Simulated time, $\tilde{t}$')
plt.ylabel(r'Droplet Number, $N_d$')

for i in range(len(psi0_range)):
    NC_av=Profiles_NC_Average[i]
    NC_up=np.array(NC_av)+np.array(Profiles_NC_Std[i])
    NC_down=np.array(NC_av)-np.array(Profiles_NC_Std[i])
    
    plt.plot(t_range,NC_av,marker='o',markersize=1.5,alpha=0.50,linestyle='-',label=str('{0:.2f}'.format(psi0_range[i])))
    #Include error bars
    plt.fill_between(t_range, NC_up, NC_down,color=plt.gca().lines[-1].get_color(), alpha=0.2)

plt.legend(title=r'$\psi_0$',loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)

plt.figure()
plt.minorticks_on()
plt.xlabel(r'Simulated time, $\tilde{t}$')
plt.ylabel(r'Average Diameter, $\tilde{D}_\text{av}$')

for i in range(len(psi0_range)):
    DS_av=Profiles_DS_Average[i]
    DS_up=np.array(DS_av)+np.array(Profiles_DS_Std[i])
    DS_down=np.array(DS_av)-np.array(Profiles_DS_Std[i])
    
    plt.plot(t_range,DS_av,marker='o',markersize=1.5,alpha=0.50,linestyle='-',label=str('{0:.2f}'.format(psi0_range[i])))
    #Include error bars
    plt.fill_between(t_range, DS_up, DS_down,color=plt.gca().lines[-1].get_color(), alpha=0.2)

plt.legend(title=r'$\psi_0$',loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.ylim(3,8.5)
plt.xlim(0,500)

#%% Histograms 

#Access the color-cycle 
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

#Selected time (Percentage)
Perc=50
plt.figure()
plt.minorticks_on()
plt.xlabel(r'Droplet Diameter, $\tilde{D}$')
plt.ylabel(r'Normalised Probability Density')

for i in range(len(psi0_range)):
    Histogram_Data_Full=Profiles_DS_Cumulative[i][Perc]
    #Accumulate all data from different simulations in a single list
    Histogram_Data=[]
    for j in range(len(Histogram_Data_Full)):
        Histogram_Data=np.concatenate((Histogram_Data,Histogram_Data_Full[j]))
    av,std=norm.fit(Histogram_Data)
    plt.hist(Histogram_Data,bins=25,range=(np.min(Histogram_Data),np.max(Histogram_Data)),alpha=0.55,rwidth=0.90,density='True',color=color_cycle[i])
    plt.plot(np.arange(0,20,0.01), norm.pdf(np.arange(0,20,0.01), av, std),color=color_cycle[i],label=str('{0:.2f}'.format(psi0_range[i])))

plt.legend(title=r'$\psi_0$')

plt.title(r'$\tilde{t}=$'+str(t_range[Perc]))
  
#%%Make plot with stacked vertical histograms 

#Time (expressed in %)
T=[15,50,75,100]
T=np.array(T)-1

#Figure size
W=8.5/2.54
H=9/2.54
FS=[W,H] 

#Figure frame 
fig, axes = plt.subplots(nrows=len(T), ncols=1, sharex=True,figsize=FS)
plt.subplots_adjust(hspace=0.04)

plt.minorticks_on()
plt.xlabel(r'Droplet diameter, $\tilde{D}$')
fig.text(0.04, 0.5, 'Normalised Probability Density', va='center', rotation='vertical')

for i in range(len(T)):

    #Fit Gaussian distributions
    Distribution_Colloids_Full=Profiles_DS_Cumulative[5][T[i]]
    Distribution_NoColloids_Full=Profiles_DS_Cumulative[0][T[i]]
    Distribution_Colloids=[]
    Distribution_NoColloids=[]
    for j in range(len(Distribution_Colloids_Full)):
        Distribution_Colloids=np.concatenate((Distribution_Colloids, Distribution_Colloids_Full[j]))
        Distribution_NoColloids=np.concatenate((Distribution_NoColloids, Distribution_NoColloids_Full[j]))
    
    av_Colloids,std_Colloids=norm.fit(Distribution_Colloids)
    av,std=norm.fit(Distribution_NoColloids)
    
    #Plot Gaussian distributions and histograms 
    D_range=np.arange(0,20,0.01)
    
    axes[i].plot(D_range, norm.pdf(D_range, av_Colloids, std_Colloids),color=color_cycle[i])
    axes[i].plot(D_range, norm.pdf(D_range, av, std),color=color_cycle[i])
    
    axes[i].hist(Distribution_Colloids,bins=20,range=(np.min(Distribution_Colloids),np.max(Distribution_Colloids)),alpha=0.55,rwidth=0.90,density='True',color=color_cycle[i])
    axes[i].hist(Distribution_NoColloids,bins=20,range=(np.min(Distribution_NoColloids),np.max(Distribution_NoColloids)),alpha=0.55,rwidth=0.90,density='True',color=color_cycle[i],histtype='step') 
    
    #Remove the labels of the ticks on the y-axis
    axes[i].set_yticklabels([])

#Make time-labels
Text_ys=[0.220,0.415,0.5975,0.7875][::-1]

for i in range(len(Text_ys)):

    fig.text(0.70, Text_ys[i], r'$\tilde{t}=$'+str(int(t_range[T[i]])), va='center',color=color_cycle[i])
