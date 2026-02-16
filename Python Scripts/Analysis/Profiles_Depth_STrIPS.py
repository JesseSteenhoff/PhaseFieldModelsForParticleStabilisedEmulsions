# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 19:34:12 2025

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


#%% Define functions required for morphological analysis 

#Function that extends the image with 0-values in both directions with a distance d 
def Extend_Image(I,d):
    Frame=np.zeros((I.shape[0]+2*d,I.shape[1]+2*d))
    Frame[d:Frame.shape[0]-d,d:Frame.shape[1]-d]=I
    return Frame

# Function that locates the oil-channels (Value 1) in a certain horizontal cross-section (C) of the image. Returns the total number of locate pores, their lenghts in the x-direction and the indices of their centres. 
def Find_Oil(C):
    
    #List with the sizes of the oil-channels in de x-direction
    PoreSize_x=[]
    #List with the indices of the (approximate) centre of each oil-channel
    Centre_x=[]
    
    #Counter for the size of the oil-channels
    p=0
    for i in range(len(C)):
        if C[i]==1:
            p+=1
            if C[i+1]==0:
                PoreSize_x.append(p)
                Centre_x.append(i-int(p/2))
                p=0
    return (len(PoreSize_x),PoreSize_x,Centre_x)

#Function that uses the list of x-coordinates of the oil-channel centres to find a corresponding pore-dimension in the y-direction. Als requires the y-value of the cross-section at which the channels were found.
def Calc_Oil_y(C,X,y):
    #List for the the pore-dimensions extending above the channel centre
    PoreSize_y1=[]
    #List for the the pore-dimensions extending below the channel centre
    PoreSize_y2=[]
    
    #Counter for the pore-dimension extending above the channel centre
    p_top=0
    #Counter for the pore-dimension extending below the channel centre. Starts at -1 to prevent double counting of the channel centre. 
    p_bottom=-1
    
    #For each located channel in a cross-section, determine the dimensions of the pore in the y-direction by summing the distances from the pore centre (x,y) to the nearest interface in opposite directions 
    for x in X:
        for i in range(y):
            if C[y-i,x]==1:
                p_top+=1
                if C[y-i-1,x]==0:
                    PoreSize_y1.append(p_top)
                    p_top=0
                    break    
        for i in range(C.shape[0]-y):
            if C[y+i,x]==1:
                p_bottom+=1
                if C[y+i+1,x]==0:    
                    PoreSize_y2.append(p_bottom)
                    p_bottom=-1
                    break
    
    PoreSize_y=np.array(PoreSize_y1)+np.array(PoreSize_y2)
    
    return (PoreSize_y,np.mean(PoreSize_y))

#%% Import the required data

#Range of initial nanoparticle concentrations 
psi0=[0.10,0.20,0.30,0.40,0.50]

#Total number of morphologies for each value of psi0
N_morph=25

#List with all depth profiles 
Profiles=[]

#List with the average depth profiles
Profiles_Average=[]

#List with the standard deviation in the depth profiles
Profiles_Std=[]

#For each value of psi0, extract all depth profiles and the average profile
for psi in psi0:
    
    Path0='<Path to Profiles_Domains_STrIPS Folder>'+'\psi0='+str('{0:.2f}'.format(psi))

    #Pore size profiles for each value of psi 
    Ds=[]

    for i in range(N_morph):
        
        Path=Path0+'\k=1.00\Morphology_Liquid_'+str(i)+'.npy'
        
        Morphology=np.load(Path)
        
        #Calculate the pore-size profile 
        Image=Morphology
        
        #Image dimensions 
        Ny=Image.shape[0]
        
        #Binarise the image
        Image[Image>=0.50]=1
        Image[Image<0.50]=0
        
        #Extend the image on all sides with 0-values. 
        E=10
        Image=Extend_Image(Image,E)
        
        # List with mean pore dimensions in the x-direction for each cross-section 
        PoreSize_dx=[]      
        # List with mean pore dimensions in the y-direction for each cross-section 
        PoreSize_dy=[]    
        
        #Set the range of horizontal cross-sections 
        dslice=1
        #Slices are chosen such that the regions of introduced 0-values due to image extension are not measured
        Slices=np.arange(E,Image.shape[0]-E,dslice)
        
        #Determine the pore dimensions in both x- and y-directions for each horizontal cross-section 
        for i in Slices:
            
            #Determines the number (N), x-dimensions (dx) and centre-coordinates (Xs) of the oil channels 
            N,dx,Xs=Find_Oil(Image[i])
            
            #If no oil-channels are found, dx (and therefore dy) are returned as 0 rather than NaN. 
            if np.isnan(np.mean(dx))==True:
                PoreSize_dx.append(0)
                PoreSize_dy.append(0)
            else:
                PoreSize_dx.append(np.mean(dx))
                PoreSize_dy.append(Calc_Oil_y(Image, Xs, i)[1])
        
        
        #Calculates the a measure for the pore size by averaging the determined dimensions in perpendicular directions 
        PoreSize=(np.array(PoreSize_dx)+np.array(PoreSize_dy))/2
        
        #Slices are re-scaled as to match the actual image dimensions, not the extended one
        Slices=Slices-E
        
        #Normalise pore sizes with respect to the system dimensions  
        #PoreSize=PoreSize/Ny
        
        #Normalise the distance within the image with respect to the system dimensions  
        #Slices=Slices/Ny

        #Store the pore size profile
        Ds.append((PoreSize))
    
    #Calculate the average profile and its standard deviation 
    
    #Create list with all determined pore sizes at each depth
    Ds_Total=[ [] for _ in range(len(Ds[0])) ]
    
    for i in range(N_morph):
        for j in range(len(Ds[0])):
            Ds_Total[j].append(Ds[i][j])
    
    #Calculate the average profile and the standard deviation 
    Profile_Average=[]
    Profile_Std=[]
    
    for i in range(len(Ds_Total)):
        Profile_Average.append(np.average(Ds_Total[i]))
        Profile_Std.append(np.std(Ds_Total[i]))
    
    Profiles_Average.append(Profile_Average)
    Profiles_Std.append(Profile_Std)
    
    Profiles.append(Ds)
    

#%% Select regions for visualisation  

#Normalised depth
Depth=np.array(Slices)/len(Profiles_Average[0])  

plt.figure()
plt.minorticks_on()
plt.xlabel(r'Depth, $\tilde{y}/\tilde{L}$')
plt.ylabel(r'Domain size, $\tilde{d}$')
    
#Choose region (%)
Start=5
End=95

Domain=np.arange(Start/100,End/100,0.01)

for i in range(len(psi0)):
    
    X=Depth[int(len(Depth)*Start/100):int(len(Depth)*End/100)]
    Y=Profiles_Average[i][int(len(Depth)*Start/100):int(len(Depth)*End/100)]
    Err=Profiles_Std[i][int(len(Depth)*Start/100):int(len(Depth)*End/100)]
    
    Yup=np.array(Y)+np.array(Err)
    Ybot=np.array(Y)-np.array(Err)
    
    plt.plot(X,Y,marker='o',markersize=1.5,alpha=0.50,linestyle='',label=str('{0:.2f}'.format(psi0[i])))
    #Include error bars
    plt.fill_between(X, Yup, Ybot,color=plt.gca().lines[-1].get_color(), alpha=0.2)

plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.,title='$\psi_0$')

#%% Select regions and perform a linear fit 

def Model(x,a,b):
    return a*x+b

plt.figure()
plt.minorticks_on()
plt.xlabel(r'Depth, $\tilde{y}/\tilde{L}$')
plt.ylabel(r'Domain size, $\tilde{d}$')
    
#Perform linear fit through selected region (%)
Start=2
End=95

End_fits=[15,40,40,40,42]

#Fit data (slopes,errors in slopes)

Slopes1=[]
Slopes2=[]

Errors1=[]
Errors2=[]

#Normalise the distance within the image with respect to the system dimensions  
Slices_Norm=Slices/Ny

for i in range(len(psi0)):
    
    X=Slices_Norm[int(len(Slices_Norm)*Start/100):int(len(Slices_Norm)*End/100)]
    Y=Profiles_Average[i][int(len(Slices_Norm)*Start/100):int(len(Slices_Norm)*End/100)]
    
    Err=Profiles_Std[i][int(len(Depth)*Start/100):int(len(Depth)*End/100)]
    
    Yup=np.array(Y)+np.array(Err)
    Ybot=np.array(Y)-np.array(Err)
    
    plt.plot(X,Y,marker='o',markersize=1.5,alpha=0.20,linestyle='')
    plt.fill_between(X, Yup, Ybot,color=plt.gca().lines[-1].get_color(), alpha=0.1)
    
    Xfit=Slices_Norm[int(len(Slices_Norm)*Start/100):int(len(Slices_Norm)*End_fits[i]/100)]
    Yfit=Profiles_Average[i][int(len(Slices_Norm)*Start/100):int(len(Slices_Norm)*End_fits[i]/100)]
        
    Domain=np.arange(np.min(Xfit),np.max(Xfit),0.01)
    
    Params,cov=curve_fit(Model, Xfit, Yfit)
    Error=np.sqrt(np.diag(cov))
    
    Slopes1.append(Params[0])
    Errors1.append(Error[0])
    
    Fit=Model(Domain,Params[0],Params[1])
    
    plt.plot(Domain,Fit,linestyle='--',color=plt.gca().lines[-1].get_color(),label=str('{0:.2f}'.format(psi0[i])))
    
    Xfit2=Slices_Norm[int(len(Slices_Norm)*End_fits[i]/100):int(len(Slices_Norm)*End/100)]
    Yfit2=Profiles_Average[i][int(len(Slices_Norm)*End_fits[i]/100):int(len(Slices_Norm)*End/100)]
    
    Params2,cov2=curve_fit(Model, Xfit2, Yfit2)
    Error2=np.sqrt(np.diag(cov2))
    
    Slopes2.append(Params2[0])
    Errors2.append(Error2[0])
    
    Domain2=np.arange(np.min(Xfit2),np.max(Xfit2),0.01)
    
    Fit2=Model(Domain2,Params2[0],Params2[1])
    
    plt.plot(Domain2,Fit2,linestyle='--',color=plt.gca().lines[-1].get_color())
 
    
plt.ylim(6.5,28)
plt.legend(title='$\psi_0$',ncol=2,fontsize=8)


#%% Perform linear fits over all profiles, find error in the fitted slopes

Slopes1_Average=[]
Slopes2_Average=[]

Slopes1_Std=[]
Slopes2_Std=[]

for i in range(len(psi0)):
    Slopes1=[]
    Slopes2=[]
    for j in range(N_morph):
        
        Xfit=Slices[int(len(Slices)*Start/100):int(len(Slices)*End_fits[i]/100)]
        Yfit=Profiles[i][j][int(len(Slices)*Start/100):int(len(Slices)*End_fits[i]/100)]
            
        Params,cov=curve_fit(Model, Xfit, Yfit)
        
        Slopes1.append(Params[0])
        
        
        Xfit2=Slices[int(len(Slices)*End_fits[i]/100):int(len(Slices)*End/100)]
        Yfit2=Profiles[i][j][int(len(Slices)*End_fits[i]/100):int(len(Slices)*End/100)]
        
        Params2,cov2=curve_fit(Model, Xfit2, Yfit2)
        
        Slopes2.append(Params2[0])
    
    Slopes1_Average.append(np.average(Slopes1))
    Slopes2_Average.append(np.average(Slopes2))
    
    Slopes1_Std.append(np.std(Slopes1))
    Slopes2_Std.append(np.std(Slopes2))


plt.figure()
plt.xlabel(r'Initial nanoparticle concentration, $\psi_0$')
plt.ylabel(r'Slope, $\Delta\tilde{d}/ \Delta\tilde{y}$')

plt.errorbar(psi0,Slopes1_Average,Slopes1_Std,linestyle='--', marker='o',markersize=3,capsize=3,label='Region I')
plt.errorbar(psi0,Slopes2_Average,Slopes2_Std,linestyle='--', marker='o',markersize=3,capsize=3,label='Region II')

plt.legend(frameon='')


#%% Select regions and perform a Polynomial Fit 

plt.figure(figsize=(8.4695/2.54,6.4008/2.54))
plt.minorticks_on()
plt.xlabel(r'Depth, $\tilde{y}/\tilde{L}$')
plt.ylabel(r'Domain size, $\tilde{d}$')
    
#Perform linear fit through selected region (%)
Start=2
End=95

#Normalise the distance within the image with respect to the system dimensions  
Slices_Norm=Slices/Ny

#Degree of Polynomial
Deg=3

#Storage list for Fits
Fits=[]

for i in range(len(psi0)):
    
    X=Slices_Norm[int(len(Slices_Norm)*Start/100):int(len(Slices_Norm)*End/100)]
    Y=Profiles_Average[i][int(len(Slices_Norm)*Start/100):int(len(Slices_Norm)*End/100)]
    
    Err=Profiles_Std[i][int(len(Depth)*Start/100):int(len(Depth)*End/100)]
    
    Yup=np.array(Y)+np.array(Err)
    Ybot=np.array(Y)-np.array(Err)
    
    plt.plot(X,Y,marker='o',markersize=1.5,alpha=0.20,linestyle='')
    plt.fill_between(X, Yup, Ybot,color=plt.gca().lines[-1].get_color(), alpha=0.1)
    
    Xfit=Slices_Norm[int(len(Slices_Norm)*Start/100):int(len(Slices_Norm)*End/100)]
    Yfit=Profiles_Average[i][int(len(Slices_Norm)*Start/100):int(len(Slices_Norm)*End/100)]
        
    Domain=np.arange(np.min(Xfit),np.max(Xfit),0.01)
    
    Fit=np.polynomial.polynomial.Polynomial(np.polynomial.polynomial.polyfit(Xfit,Yfit,deg=Deg))
    Fits.append(Fit)

    plt.plot(Domain,Fit(Domain),linestyle='--',color=plt.gca().lines[-1].get_color(),label=str('{0:.2f}'.format(psi0[i])))
      
#plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.,title='$\psi_0$')
plt.ylim(6.5,29)
plt.legend(title='$\psi_0$',ncol=2,fontsize=8)

