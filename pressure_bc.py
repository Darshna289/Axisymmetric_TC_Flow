import numpy as np
from globals import omega, r_cyl, nu, dr, dz, n_r, n_z, ro

def pressure_bc(P):
    
    #Apply boundary conditions to a 2D pressure field.

    #fixing top outer corner as zero
    P[:,-1]=P[:,-2]
    P[0,:]=P[1,:]
    P[:,0]=P[:,1]
    P[-1,:]=P[-2,:]

    P[0,-1]=0
 
    return P 
