import numpy as np
from globals import omega, r_cyl, nu, dr, dz, n_r, n_z, ro

def compute_divergence(u_r, u_z, r_cyl, dr, dz):
    
    div = np.zeros((n_r, n_z))

    for i in range(1, n_r - 1):  #till n_r-2 becuase r_cyl[i]
          for j in range(1, n_z - 1):
            r_I = (r_cyl[i]+r_cyl[i+1])/2

            # ∂(r u_r)/∂r ≈ (r_{i+1} u_r[i+1,j] - r_{i-1} u_r[i,j]) / (2dr) #incorrect in the context of staggered grid
            d_ru_r_dr = (((r_cyl[i]+r_cyl[i+1]) * u_r[i, j])/2 - ((r_cyl[i]+r_cyl[i-1]) * u_r[i-1, j])/2) /dr
            #d_ru_r_dr = (((r_cyl[i+1]+r_cyl[i+2]) * u_r[i+1, j])/2 - ((r_cyl[i]+r_cyl[i-1]) * u_r[i-1, j])/2) /(2*dr) #second order discretization

            # ∂u_z/∂z ≈ (u_z[i, j+1] - u_z[i, j-1]) / (2dz) #incorrect in the context of staggered grid
            duz_dz = (u_z[i, j] - u_z[i, j - 1]) / (dz)
            #\duz_dz = (u_z[i, j+1] - u_z[i, j - 1]) / (2*dz) #second order discretization

            # full divergence
            div[i, j] = (1/r_I) * d_ru_r_dr + duz_dz
        
    return div