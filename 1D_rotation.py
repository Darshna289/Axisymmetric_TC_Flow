import numpy as np
import matplotlib.pyplot as plt

# Constants
r_o = 1.0
r_i = 0.9
nu=1e-6

n_theta = 50
n_r = 50

# Grid spacing
r = np.linspace(r_i, r_o, n_r)
theta = np.linspace(0, 2 * np.pi, n_theta)
dr=(r_o-r_i)/n_r
dtheta=(2*np.pi)/n_theta

# Create 2D meshgrid
R, Theta = np.meshgrid(r, theta, indexing='ij')

# Convert polar to Cartesian coordinates
X = R * np.cos(Theta)
Y = R * np.sin(Theta)


'''
# Plot mesh lines
plt.figure(figsize=(6, 6))
for i in range(n_r):  # radial lines
    plt.plot(X[i, :], Y[i, :], color='blue', linewidth=0.5)
for j in range(n_theta):  # angular lines
    plt.plot(X[:, j], Y[:, j], color='blue', linewidth=0.5)

plt.title('Mesh Between Inner and Outer Cylinder')
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
'''

#function to calculate value at a particular time step
def soln(ri,u_theta_inst,i):
    du_dr2=(u_theta_inst[i+1]-2*u_theta_inst[i]+u_theta_inst[i-1])/dr**2
    du_dr=(u_theta_inst[i+1]-u_theta_inst[i-1])/(2*dr)

    fn=nu*(du_dr2+du_dr/ri-(u_theta_inst[i])/(ri**2))
    return fn

#initializing
u_theta=np.zeros((n_r))
u_theta_o=np.zeros((n_r))
u_theta_oo=np.zeros((n_r))

T=4000
dt=0.8

n_t=int(T/dt)

for t_step in range(1,n_t):
    print("time step:",t_step)
    t=dt*t_step

    u_theta_oo=u_theta_o.copy()
    u_theta_o=u_theta.copy()

    # initial u_theta at the wall of inner cylider
    u_theta[0]=10
    u_theta_o[0]=10
    u_theta_oo[0]=10

    if t_step==1:
            for i in range(1,n_r-1):
                ri=r[i]
                print(soln(ri,u_theta_o,i))
                u_theta[i]=u_theta_o[i]+dt*soln(ri,u_theta_o,i)
        
    else:
        for i in range(1,n_r-1):
            ri=r[i]
            u_theta[i]=u_theta_o[i]+dt*(1.5*soln(ri,u_theta_o,i)-0.5*soln(ri,u_theta_oo,i))
    
    u_theta[-1]=0
    u_theta_o[-1]=0
    u_theta_oo[-1]=0

    #plt.clf()
    #plt.plot(u_theta,"o-")
    #plt.pause(0.1)

u_ana=(10*r_i/(r_i**2-r_o**2))*r-(10*r_i*r_o**2/(r_i**2-r_o**2))/r  # steady state analytical solution

u_theta_az=np.zeros((n_r,n_theta))
for k in range(n_theta):
    u_theta_az[:,k]=u_theta

plt.plot(r,u_theta,"o-")
plt.plot(r,u_ana,"-")
plt.xlabel("r")
plt.ylabel("U_theta")
plt.legend(["numerical","analytical"])
plt.show()

plt.figure(figsize=(6,6))
plt.pcolormesh(X, Y, u_theta_az, shading='auto', cmap='viridis')
plt.colorbar(label=r'$u_\theta$')
plt.title(r'2D Colormap of $u_\theta(r)$')
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()