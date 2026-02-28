import numpy as np

# Physical parameters
h=1
w=h/20
r_in=w*(1/(1/0.91-1))

ro=r_in/0.91
nu=1e-6 #viscosity

Re=1600
omega = Re*nu/(w*r_in)

# Numerical parameters
n_r = 10
dr = (ro - r_in) / (n_r - 1)
dz=dr/0.4       # to maintain a minimum aspect ratio in mesh elements
n_z=int(h/dz)+1
#dz = h / (n_z - 1)

# Create the grid
r = np.linspace(r_in, ro, n_r)
z = np.linspace(0, h, n_z)

R, Z = np.meshgrid(r, z)

r_cyl = np.linspace(r_in, ro, n_r)
