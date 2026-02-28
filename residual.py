import numpy as np
from globals import omega, r_cyl, nu, dr, dz, n_r, n_z, ro


def residual(P, rhs, dr, dz):
    """
    Compute the residual of a finite difference equation.
    
    Parameters:
    P    : np.ndarray
        Solution matrix (nx, ny)
    rhs  : np.ndarray
        Right-hand side matrix (nx, ny)
    dx   : float
        Grid spacing in the x-direction
    dy   : float
        Grid spacing in the y-direction
    
    Returns:
    resid : np.ndarray
        Residual matrix (nx, ny)
    """
    nx, ny = P.shape
    resid = np.zeros((nx, ny))
    # Compute residual in the interior grid points (excluding boundaries)
    resid[1:-1, 1:-1] = (
        rhs[1:-1, 1:-1] 
        - (P[2:, 1:-1] - 2 * P[1:-1, 1:-1] + P[:-2, 1:-1]) / dr**2
        - (P[1:-1, 2:] - 2 * P[1:-1, 1:-1] + P[1:-1, :-2]) / dz**2
        -(P[2:, 1:-1]-P[:-2, 1:-1])*ro/dr
    )
    return resid