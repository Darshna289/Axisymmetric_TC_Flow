import numpy as np
from globals import omega, r_cyl, nu, dr, dz, n_r, n_z, ro

def uz_eval(I,J,ur,uz):
    i=I
    j=J
    rJ=r_cyl[i]
    urduz_dr=(((ur[I,j]+ur[I,j+1])/2)*((uz[i,J]+uz[i+1,J])/2) - ((ur[I-1,j]+ur[I-1,j+1])/2)*((uz[i-1,J]+uz[i,J])/2))/dr
    uzduz_dz=(((uz[i,J]+uz[i,J+1])/2)**2-((uz[i,J]+uz[i,J-1])/2)**2)/dz
    d2uz_dz2=(uz[i,J+1]-2*uz[i,J]+uz[i,J-1])/dz**2
    duz_dr=(uz[i+1,J]-uz[i-1,J])/2*dr
    d2uz_dz2=(uz[i+1,J]-2*uz[i,J]+uz[i-1,J])/dr**2

    fz=-urduz_dr-uzduz_dz+nu*(d2uz_dz2+duz_dr/rJ+d2uz_dz2)
    return fz