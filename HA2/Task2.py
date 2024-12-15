import torch
import matplotlib.pyplot as plt
from torch.autograd import grad

# torch settings
torch.set_default_dtype(torch.float64)  # set default tensor type
torch.set_num_threads(8)  # adapt to your machine
# matplotlib settings
plt.rcParams["text.usetex"] = True  # use latex for font rendering
plt.rcParams["lines.markersize"] = 3  # set default marker size
plt.rcParams["font.size"] = 18  # set default font size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # use GPU if available

# Material parameters
mu = 384.614  # Pa
lamda = 576.923  # Pa 

# initialize the displacement field
Lx = 1 # sizing parameter in x-direction
Ly = 2 # sizing parameter in y-direction
Nx = 80 * Lx + 1 # number of grid points in x-direction
Ny = 80 * Ly + 1 # number of grid points in y-direction
shape1 = ( Nx , Ny ) # shape of the grid
Nx = torch.tensor( Nx ) # convert to tensor
Ny = torch.tensor( Ny ) # convert to tensor
dx = ( Lx / ( Nx - 1) ) # grid spacing in x-direction
dy = ( Ly / ( Ny - 1) ) # grid spacing in y-direction
X = torch.meshgrid(torch.linspace(0 , Lx , Nx ) , torch.linspace(0 , Ly ,Ny ) , indexing ='ij') # create grid
X = torch.cat(( X[0].reshape( -1 , 1) , X[1].reshape( -1 , 1) ) , dim =1) # reshape grid to 2D array
X.requires_grad_(True) # enable gradients for X
# define displacement field 'u' as a function of the grid points 'X'
u = torch.stack((torch.sin(2 * X[:, 0]) + 0.5 * X[:, 1], 0.3 * torch.cos(1.5 * X[:, 0]) * X[:, 1]**2 ), dim=1)
u = torch.stack((0.2 * X[:, 0] * torch.cos(torch.pi * X[:, 1]), 
                       0.2 * X[:, 1] * torch.sin(2 * torch.pi * X[:, 0])), dim=1)
u = torch.stack((0.2 * torch.sin(0.5 * torch.pi * X[:, 0]) * torch.cos(0.5 * torch.pi * X[:, 1]), 0.2 * torch.cos(0.5 * torch.pi * X[:, 0]) * torch.sin(0.5 * torch.pi * X[:, 1])), dim=1)

                       
# calculate the deformation gradient using autograd.grad

duxdX = grad(u[:, 0].unsqueeze(1) , X , torch.ones(X.size()[0] , 1, device=device) , create_graph = True, retain_graph=True)[0] # compute derivatives of u[:, 0] w.r.t. X
# Compute deformation gradient F using autograd.grad
F = torch.zeros((X.shape[0], 2, 2), dtype=torch.float64)  # Initialize deformation gradient
for i in range(2):  # Loop over dimensions
    for j in range(2):  # Compute derivatives of u[:, i] w.r.t. X[:, j]
        duxdX = grad(u[:, i].unsqueeze(1) , X , torch.ones(X.size()[0] , 1, device=device) , create_graph = True, retain_graph=True)[0][:, j]
        F[:, i, j] = duxdX + (1 if i == j else 0)  # Add identity for deformation gradient

# Compute determinant of F (Jacobian J)
# for 2x2 matrix, determinant is given by ad - bc
J = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]

# Compute first invariant I1 = F:F with einsum
I1 = torch.einsum('bik,bik->b', F, F)

# Helmholtz free energy density Ψ according to the assignment
psi = lamda / 2 * (torch.log(J))**2 - mu * torch.log(J) + mu / 2 * (I1 - 2)

# Plot the results
fig, axs = plt.subplots(2, figsize=(8, 10))  # create figure and axes
# Plot the undeformed grid
axs[0].scatter(X[:, 0].detach(), X[:, 1].detach(), color='black', label='Reference Configuration')
axs[0].set_title("Reference Configuration")
axs[0].legend()

# Plot the deformed grid and energy distribution
sc = axs[1].scatter((X[:, 0] + u[:, 0]).detach(), (X[:, 1] + u[:, 1]).detach(),
               c=psi.detach(), cmap='viridis', vmin=psi.min(), vmax=psi.max())
axs[1].set_title("Deformed Configuration with Energy Distribution")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")
fig.colorbar(sc, ax=axs[1], label='Helmholtz Free Energy Density $\Psi$')


plt.tight_layout()  # adjust layout
plt.show()
