import numpy as np
from globals import omega, r_cyl, nu, dr, dz, n_r, n_z, ro


def ur_eval(I,J,ur,uz,u_th):
    i=I
    j=J
    rI=(r_cyl[i]+r_cyl[i+1])/2
    urdur_dr=(((ur[I,j]+ur[I+1,j])/2)**2 - ((ur[I,j]+ur[I-1,j])/2)**2)/dr
    u_th_I=(u_th[i,j]+u_th[i+1,j])/2
    uzdur_dz=(((uz[i,J]+uz[i+1,J])/2)*((ur[I,j]+ur[I,j+1])/2) - ((uz[i,J]+uz[i-1,J])/2)*((ur[I-1,j]+ur[I-1,j+1])/2))/dz
    d2ur_dr2=(ur[I+1,j]-2*ur[I,j]+ur[I-1,j])/dr**2
    dur_dr=(ur[I+1,j]-ur[I-1,j])/(2*dr)
    d2ur_dz2=(ur[I,j+1]-2*ur[I,j]+ur[I,j-1])/dz**2

    fr= -urdur_dr+u_th_I**2/rI-uzdur_dz+nu*(d2ur_dr2+dur_dr/rI-ur[I,j]/rI**2+d2ur_dz2)
    return fr