import numpy as np
from globals import omega, r_cyl, nu, dr, dz, n_r, n_z, ro

def u_th_eval(i,j,ur,uz,uth):
    I=i
    J=j
    ri=r_cyl[i]
    ur_iminus1=(ur[I-1,j]+ur[I-2,j])/2
    ur_iplus1=(ur[I,j]+ur[I+1,j])/2
    ur_i=(ur[I,j]+ur[I-1,j])/2
    uz_jplus1=(uz[i,J]+uz[i,J+1])/2
    uz_jminus1=(uz[i,J]+uz[i,J-1])/2

    urduth_dr=(ur_iplus1*uth[i+1,j]-ur_iminus1*uth[i-1,j])/(2*dr)
    uzduth_dz=(uz_jplus1*uth[i,j+1]-(uz_jminus1*uth[i,j-1]))/(2*dz)
    d2uth_dr2=(uth[i+1,j]-2*uth[i,j]+uth[i-1,j])/(dr**2)
    duth_dr=(uth[i+1,j]-uth[i-1,j])/(2*dr)
    d2uth_dz2=(uth[i,j+1]-2*uth[i,j]+uth[i,j-1])/(dz**2)

    fth=-urduth_dr-ur_i*uth[i,j]/ri-uzduth_dz+nu*(d2uth_dr2+duth_dr/ri-uth[i,j]/ri**2+d2uth_dz2)
    return fth