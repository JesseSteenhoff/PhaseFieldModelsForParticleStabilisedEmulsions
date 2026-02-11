# Python Script Overview: Phase-Field Simulations

**CahnHilliard_Colloids_1D**: Phase-field simulations that solve coupled Cahn-Hilliard equations of liquids and colloids for a 1D interface. It also calculates tracks both the interfacial tension and excess over the course of the simulation. 

**DynamicEquations_1D**: Phase-field simulations that solve dynamic equations of liquids and colloids for a 1D interface. 

**DynamicEquations_2D**: Phase-field simulations that solve dynamic equations of liquids and colloids for a 2D system with periodic boundary conditions. 

**STrIPS_2D**: Phase-field simulations of bijel formation via solvent-transfer induced phase separation in a 2D system (ambient liquid phase on top, solid substrate on bottom, periodic conditions along horizontal directions). 

**STrIPS_3D**: Phase-field simulations of bijel formation via solvent-transfer induced phase separation in a 3D system (ambient liquid phase on top, solid substrate on bottom. periodic conditions in-plane). 

Most of these scripts make use of standarrd libraries such as Numpy and Scipy. For visualisation they make use of the SciencePlots library (see https://pypi.org/project/SciencePlots/ for installation instructions). The Numba compiler is used to increase the performance of the STrIPS simulations (see https://numba.pydata.org for instructions for installation and general use). 


