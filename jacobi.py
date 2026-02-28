import numpy as np
from globals import omega, r_cyl, nu, dr, dz, n_r, n_z, ro

def jacobi(P, rhs, dr, dz):
    """
    Perform one iteration of the Jacobi method for solving a 2D finite difference system.

    Parameters:
    P   : np.ndarray
        Initial temperature field (nx, ny)
    rhs : np.ndarray
        Right-hand side forcing term (nx, ny)
    dx  : float
        Grid spacing in the x-direction
    dy  : float
        Grid spacing in the y-direction

    Returns:
    P_next : np.ndarray
        Updated temperature field after one Jacobi iteration
    """
    nx, ny = P.shape
    P_next = P.copy()  # Copy to avoid modifying the input during iteration

    for i in range(1, nx-1):
        ri=r_cyl[i]
        for j in range(1, ny-1):
            P_next[i, j] = (rhs[i,j]+P[i+1,j]*(-1/(ri*dr)-1/dr**2)+P[i-1,j]*(1/(ri*dr)-1/dr**2)+(P[i,j+1]+P[i,j-1])*(-1/dz**2))*((-2/dr**2 - 2/dz**2)**(-1))   

    return P_next
