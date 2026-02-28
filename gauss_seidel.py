import numpy as np
from globals import omega, r_cyl, nu, dr, dz, n_r, n_z, ro

def gauss_seidel(P, rhs, dr, dz):
    """
    Perform Gauss-Seidel iteration with relaxation for a 2D finite difference grid.

    Parameters:
    P     : np.ndarray
        Initial temperature field (nx, ny)
    rhs   : np.ndarray
        Right-hand side (nx, ny)
    dx    : float
        Grid spacing in x-direction
    dy    : float
        Grid spacing in y-direction
    omega : float, optional
        Relaxation factor (default: 0)
        (omega < 1: under-relaxation; 1 < omega < 2: successive over-relaxation)

    Returns:
    T_next : np.ndarray
        Updated temperature field after one Gauss-Seidel iteration
    """
    nx, ny = P.shape
    P_next = P.copy()

    # Gauss-Seidel iteration
    for i in range(1, nx-1):
        ri=r_cyl[i]
        for j in range(1, ny-1):
            #P_next[i, j] = (1-RF)*P[i,j] -RF*((rhs[i,j]+P[i+1,j]*(-1/(ri*dr)-1/dr**2)+P_next[i-1,j]*(1/(ri*dr)-1/dr**2)+(P[i,j+1]+P_next[i,j-1])*(-1/dz**2))*((-2/dr**2 - 2/dz**2)**(-1)))   
            P_next[i, j] = (rhs[i,j]+P[i+1,j]*(-1/(ri*dr)-1/dr**2)+P_next[i-1,j]*(1/(ri*dr)-1/dr**2)+(P[i,j+1]+P_next[i,j-1])*(-1/dz**2))*((-2/dr**2 - 2/dz**2)**(-1))
    return P_next
