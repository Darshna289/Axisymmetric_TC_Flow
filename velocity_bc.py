import numpy as np
from globals import omega, r_in, r_cyl, nu, dr, dz, n_r, n_z, ro


def velocity_bc(ur,u_theta,uz):
    u_theta[:,0] =  2*omega*r_in- u_theta[:,1]
    u_theta[0,:] = -u_theta[1,:]
    u_theta[:,-1] = -u_theta[:,-2]
    u_theta[-1,:] = -u_theta[-2,:]

    # staggered grid for ur and uz 
    ur[:,0]= 0
    ur[0,:] = 0      
    ur[:,-1] = 0 
    ur[-1,:] = 0 

    uz[:,0]= 0
    uz[0,:] = 0
    uz[:,-1] = 0 
    uz[-1,:] = 0 

    return ur,u_theta,uz
