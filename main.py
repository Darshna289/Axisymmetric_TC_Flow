import numpy as np
import matplotlib.pyplot as plt
import logging

from velocity_bc import velocity_bc
from pressure_bc import pressure_bc
from jacobi import jacobi
from gauss_seidel import gauss_seidel
from ur_eval import ur_eval
from uz_eval import uz_eval
from u_th_eval import u_th_eval
from compute_divergence import compute_divergence
from residual import residual

from globals import omega, r_in, r_cyl, nu, dr, dz, n_r, n_z, ro


dt_min = 0.1 # A small time step to start with
dt_new=dt_min
t_max = 10

# r_cyl = np.linspace(r_in, ro, n_r)
#print(dr/dz)

# Initialize fields
u_r = np.zeros((n_r-1, n_z)) #u_r = np.zeros((n_r, n_z))        # radial velocity at r faces
u_z = np.zeros((n_r, n_z-1)) #u_z = np.zeros((n_r, n_z))        # axial velocity at z faces
u_theta = np.zeros((n_r, n_z))      # azimuthal velocity at centers

#Initialising velocity profile for u_theta
for k in range(len(r_cyl)):
    R=r_cyl[k]
    u_theta[k,0] = (omega*r_in/(r_in**2-ro**2))*R-(omega*r_in*ro**2/(r_in**2-ro**2))/R 

for j in range(n_z):
    u_theta[:,j]=u_theta[:,0]

u_r_star = np.zeros((n_r, n_z))   
u_z_star = np.zeros((n_r, n_z))  


# Initialize Previous time levels
u_r_o = u_r.copy(); u_r_oo = u_r.copy()
u_z_o = u_z.copy(); u_z_oo = u_z.copy()
u_theta_o = u_theta.copy(); u_theta_oo = u_theta.copy()

# Initialize pressure field
P = np.zeros((n_r, n_z)) 
P=pressure_bc(P)
rhs = np.zeros((n_r, n_z))

errorL2_0 = np.linalg.norm(residual(P, rhs, dr, dz))
print("L20",errorL2_0)
errorL2 = 1.0
resid_history = [errorL2]
itermax = 500
resmax  = 1.e-6

time=0
t_step=0

P2=np.zeros((n_r,n_z))
resid2=np.zeros((n_r,n_z))

# Time integration
while time<t_max:
    t_step=t_step+1
    time=time+dt_min
    print("time=",time)

#for t_step in range(1, n_step):
    #print("time step:", t_step)

    u_r,u_theta,u_z=velocity_bc(u_r,u_theta,u_z)
    
    # Shift time levels
    u_r_oo[:], u_z_oo[:], u_theta_oo[:] = u_r_o, u_z_o, u_theta_o
    u_r_o[:], u_z_o[:], u_theta_o[:] = u_r, u_z, u_theta

    if t_step==1:
        for j in range(1, n_z-2):
            for i in range(1, n_r-2):
                ri=r_cyl[i]
                #print(u_th_eval(i,j,u_r,u_z,u_theta))
                u_r_star[i,j] = u_r_o[i, j] + dt_min * ur_eval(i, j, u_r_o, u_z_o, u_theta_o)
                                       
                u_z_star[i,j] = u_z_o[i, j] + dt_min *uz_eval(i, j, u_r_o, u_z_o)
                                           
                u_theta[i,j] = u_theta_o[i, j] + dt_min * u_th_eval(i, j, u_r_o, u_z_o, u_theta_o)

        # need to add a step to calculate u_theta at nr-1,nz-1        
                                                   
    else:
        for j in range(1, n_z-2):
            for i in range(1, n_r-2):
                # Compute predicted velocities u*_r, u*_z, u*_theta
                # Use Adams-Bashforth for time stepping, central differencing for spatial
                # For brevity, denote RHS_op() as a function that computes diffusive + convective terms

                u_r_star[i,j] = u_r_o[i, j] + dt_min * (1.5 * ur_eval(i, j, u_r_o, u_z_o, u_theta_o) - 0.5 * ur_eval(i, j, u_r_oo, u_z_oo, u_theta_oo))
                u_z_star[i,j] = u_z_o[i, j] + dt_min * (1.5 * uz_eval(i, j, u_r_o, u_z_o) - 0.5 * uz_eval(i, j, u_r_oo, u_z_oo))
                u_theta[i,j] = u_theta_o[i, j] + dt_min * (1.5 * u_th_eval(i, j, u_r_o, u_z_o, u_theta_o) - 0.5 * u_th_eval(i, j, u_r_oo, u_z_oo, u_theta_oo))
                
    # Step 2: Solve Poisson's equation for pressure
    # \nabla^2 p^{n+1} = (\nabla \cdot u^*) / dt
    #rhs = compute_divergence(u_r, u_z, r_cyl, dr, dz) / dt_min
    div=compute_divergence(u_r_star,u_z_star,r_cyl,dr,dz)
    rhs = div/dt_min
      
    iter_count=0
    max_resid=1

    while errorL2 > resmax and iter_count < itermax:
        P=pressure_bc(P)
        P=gauss_seidel(P,rhs,dr,dz)
  
        # Error computation
        resid = residual(P, rhs, dr, dz)
        #print("::",t_step, resid)
        if t_step==2:
            rhs2=div/dt_min
            P2=gauss_seidel(P,rhs,dr,dz)
            resid2=residual(P2, rhs2, dr, dz)
            
        errorL2_0=np.linalg.norm(resid2)
        #print(t_step,errorL2_0)

        errorL2 = np.linalg.norm(resid) / errorL2_0
        
        #print(errorL2)
        resid_history.append(errorL2)
        iter_count=iter_count+1
        

    #print("errL2",errorL2,iter_count)
        #print(errorL2)

    #print("resi_max",np.max(resid),time)
    # print(iter_count)
    # Step 3: Correct velocities with pressure gradient
    for i in range(1, n_r-1):
        for j in range(1, n_z-1):
            u_r[i, j] =u_r_star[i,j] - dt_min * (P[i+1, j] - P[i, j]) / dr
            u_z[i, j] =u_z_star[i,j] - dt_min * (P[i, j+1] - P[i, j]) / dz
            # u_theta not corrected in pressure projection (azimuthal component decoupled in incompressible axisymmetric flow)

    u_r,u_theta,u_z=velocity_bc(u_r,u_theta,u_z)
    #print(u_r[u_r!=0])
    # Apply boundary conditions to u_r, u_z, u_theta, and p
    max_ur=np.max(abs(u_r))
    max_uz=np.max(abs(u_z))
    max_uth=np.max(abs(u_theta[1:-1,1:-1]))
    print(max_uth,max_ur,max_uz,np.max(abs(P)))
    div2=compute_divergence(u_r,u_z,r_cyl,dr,dz)
    print("div_max",np.max(div2))
    
    dt_new=0.99*0.1/(max_ur/dr+max_uz/dz+nu/dr**2+nu/dz**2+nu/r_in**2) #multiplied a 0.99 to take a smaller time step than the criteria
    if dt_new<dt_min:
        dt_min=dt_new
    dt_min=dt_new
    print("dt = ",dt_new)

    #dt_min=dt_new

    
    # if t_step%5==0:
    #     plt.clf()
    #     #plt.plot(u_theta[int(n_r/2),1:])
    #     plt.imshow(u_theta[1:-1,1:-1].T)
    #     plt.colorbar()
    #     plt.pause(0.05)
    

        #print("press=",P[5,15],dt_min)
    

  
'''
plt.imshow(u_theta[1:-1,1:-1])
plt.colorbar()
plt.show()
#plt.plot(resid_history)
#plt.show()
#plt.imshow(P)
#plt.colorbar()
#plt.plot(u_theta[:,5])
plt.show()
'''
#plt.plot((u_theta[int(n_r/2),1:])/(omega*r_in),"o-")
#plt.show()
#Pt=np.transpose(P)
#plt.imshow((u_theta))
#plt.colorbar()
#print((u_theta[int(n_r/2),:])/(omega*r_in))
#plt.show()

#plt.plot(u_r[int(n_r/2),1:-1],"o-")
#plt.imshow(u_theta[1:-1,1:-1])
#plt.imshow(u_theta)
#plt.colorbar()
#plt.show()


#calculating ur and uz on collocated grid
ur_collocated = 0.5 * (u_r[:-1, :] + u_r[1:, :])  
uz_collocated = 0.5 * (u_z[:, :-1] + u_z[:, 1:])
print(ur_collocated.shape, uz_collocated.shape)
utheta_collocated = u_theta[1:-1,1:-1]


u_magnitude = np.sqrt(ur_collocated[:,1:-1]**2 + uz_collocated[1:-1,:]**2 + utheta_collocated**2) 
print(np.mean(u_magnitude))
u_fin = u_magnitude #- np.mean(u_magnitude)

'''
# Plot heatmap and vector field
#plt.figure(figsize=(5, 10))
plt.pcolormesh((u_fin)[1:,1:].T, shading='auto', cmap='viridis')
#plt.pcolormesh(u_theta[1:-1,1:-1], shading='auto', cmap='viridis')
plt.colorbar(label='Magnitude')
plt.xlabel('r')
plt.ylabel('z')
plt.title('u_theta')
#plt.plot(u_theta[2,:],"o-")
plt.show()
'''

'''
u_ana=[]
for k in range(len(r_cyl)):
    R=r_cyl[k]
    u_ana.append((omega*r_in/(r_in**2-ro**2))*R-(omega*r_in*ro**2/(r_in**2-ro**2))/R )

u_ana[-1]=0
u_ana[0]=2*omega*r_in-u_ana[1]

print(u_theta.shape,len(u_ana))

#plt.plot(u_ana,"o-")
plt.plot(u_theta[3,:],"o-")
plt.title("u_theta")

plt.show()
'''

r = np.linspace(0, 1, ur_collocated[:,1:-1].shape[0])
z = np.linspace(0, 2, ur_collocated[:,1:-1].shape[1])
R, Z = np.meshgrid(r, z, indexing='ij')

# Extract velocities
u_r = ur_collocated[:,1:-1]
u_z = uz_collocated[1:-1,:]

# Compute magnitude
magnitude = np.sqrt(u_r**2 + u_z**2)

# Normalize vectors for uniform arrow size
epsilon = 1e-12  # to prevent division by zero
u_r_norm = u_r / (magnitude + epsilon)
u_z_norm = u_z / (magnitude + epsilon)

# Plot
# plt.figure(figsize=(8, 6))
# quiv = plt.quiver(R, Z, u_r_norm, u_z_norm, magnitude, cmap='viridis')
# plt.xlabel('Radial (r)')
# plt.ylabel('Axial (z)')
# plt.title('Velocity Direction in r-z Plane (Colored by Magnitude)')
# plt.colorbar(quiv, label='Velocity Magnitude')
# plt.axis('equal')
# plt.tight_layout()
# plt.show()

r = np.linspace(0, 1, ur_collocated[:,1:-1].shape[0])
z = np.linspace(0, 2, ur_collocated[:,1:-1].shape[1])
R, Z = np.meshgrid(r, z, indexing='xy')

# Extract velocities
u_r = ur_collocated[:,1:-1]
u_z = uz_collocated[1:-1,:]

# Compute magnitude
magnitude = np.sqrt(u_r**2 + u_z**2)


# Normalize vectors for uniform arrow size
epsilon = 1e-12  # to prevent division by zero
u_r_norm = u_r / (magnitude + epsilon)
u_z_norm = u_z / (magnitude + epsilon)

print(R.shape, u_r_norm.shape, u_z_norm.shape)

# Plot streamlines/vector plot
plt.figure(figsize=(8, 6))
#quiv = plt.quiver(R, Z, u_r_norm, u_z_norm, magnitude, cmap='viridis')
plt.streamplot(R,Z,u_r_norm.T,u_z_norm.T,color=magnitude.T)
plt.xlabel('Radial (r)')
plt.ylabel('Axial (z)')
plt.title('Velocity Direction in r-z Plane (Colored by Magnitude)')
plt.colorbar()
#plt.colorbar(quiv, label='Velocity Magnitude')
plt.axis('equal')
plt.tight_layout()
plt.show()


