import matplotlib
matplotlib.rcParams["text.usetex"] = True
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)
torch.set_num_threads(8) # Use _maximally_ 8 CPU cores
plt.rcParams.update({'font.size': 16})

device = torch.device("cpu")
#device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
device = torch.device(device)



Lx = 2
Ly = 1
samples_x = 20 * Lx + 1
samples_y = 20 * Ly + 1
delta_x = Lx / (samples_x - 1)
delta_y = Ly / (samples_y - 1)

disp_left = 0
force_right = 0.2
force_upper = 0
force_lower = 0

"""
/|-----------------------| ->
/|                       | ->
/|                       | ->
/|-----------------------| ->
"""

sample_points = torch.meshgrid(torch.linspace(0, Lx, samples_x), torch.linspace(0, Ly, samples_y), indexing='ij')
sample_points = torch.cat((sample_points[0].reshape(-1, 1), sample_points[1].reshape(-1, 1)), dim=1)
sample_points = sample_points.to(device)
sample_points.requires_grad_(True)


hidden_dim = 16
input_dim = 2
output_dim = 2


def plot_undef ():
    plt.subplot(1,1,1)
    plt.scatter(sample_points[:, 0].detach().numpy(), sample_points[:, 1].detach().numpy(), c='black', s=5)

E = 1
nu = 0.3
mu = E/(2*(1+nu))
lam = E*nu / ((1+nu)*(1-2*nu))


def geteps(X, U):
    duxdxy = torch.autograd.grad(U[:, 0].unsqueeze(1), X, torch.ones(X.size()[0], 1, device=device),
                                 create_graph=True, retain_graph=True)[0]
    duydxy = torch.autograd.grad(U[:, 1].unsqueeze(1), X, torch.ones(X.size()[0], 1, device=device),
                                 create_graph=True, retain_graph=True)[0]
    H = torch.zeros(X.size()[0], X.size()[1], X.size()[1], device=device)
    H[:, 0, :] = duxdxy
    H[:, 1, :] = duydxy
    H = H.reshape(samples_x, samples_y, 2, 2)
    eps = H
    eps[:, :, [0, 1], [1, 0]] = 0.5 * (eps[:, :, 0, 1] + eps[:, :, 1, 0]).unsqueeze(2).expand(samples_x, samples_y, 2)
    return eps

def material_model(eps):
    tr_eps = eps.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    sig = 2 * mu * eps + lam * torch.einsum('ij,kl->ijkl', tr_eps, torch.eye(eps.size()[-1], device=device))
    psi = torch.einsum('ijkl,ijkl->ij', eps, sig)
    return psi, sig

def integratePsi (psi):
    # PyTorch has a built-in function for trapezoidal integration.
    # Hence, we can use it (nested for the 2 dimensions)
    # The comparison with midpoint rule showed no
    # difference to the midpoint rule significant for the basic stability of this DEM code
    PSI = torch.trapezoid(torch.trapezoid(psi, dx = delta_y, dim=1), dx = delta_x, dim=0)
    return PSI

def calculateDivergenceLoss(sig):
    criterion = nn.MSELoss(reduction="sum")
    div_sig = (1 / (2 * delta_x) * (sig[2:, 1:-1, :, 0] - sig[:-2, 1:-1, :, 0]))
    div_sig += (1 / (2 * delta_y) * (sig[1:-1, 2:, :, 1] - sig[1:-1, :-2, :, 1]))
    div_loss = criterion(div_sig, torch.zeros(samples_x - 2, samples_y - 2, 2, device=device))
    print("div_loss: ", div_loss)
    return div_loss

def integrateTractionEnergy(U, sig):
    # Again, use built-in integration
    # Before, there was a bug: Only the farmost right points have to be integrated
    # Not U[:, :, 0], but U[-1, :, 0]
    # This bug is fixed here:
    Ty = force_right * torch.trapezoid(U[-1, :, 0], dx = delta_y)

    # Theoretically, the traction is calculated from the stress, not the force boundary condition
    # So it's up to comparison what performs better.
    #Ty = torch.trapezoid(sig[-1, :, 0, 0] * U[-1, :, 0], dx=delta_y)

    # The x-component was not necessary, so far
    # Careful: the bug was not fixed here!
    #Tx = (sig[1:-1, :, 0, 0] * U[1:-1, :, 0]).sum()
    #Tx += (sig[:, 1:-1, 1, 1] * U[:, 1:-1, 1]).sum()
    #Tx += 0.5 * (sig[[0, -1], [0, -1], 1, 1] * U[[0, -1], [0, -1], 1]).sum()
    #Tx *= delta_x

    return Ty #+ Tx

def boundaryLosses(U, sig):
    criterion = nn.MSELoss(reduction="sum")
    # New factor to multiply the losses for fixed / Dirichlet B.C.
    # The factor 100*E was too large, i.e. the other conditions were not respected in comparison.
    # The factor 1*E was too small, i.e. the B.C. was not fulfilled well
    loss_left = criterion(delta_y * U[0, :, :] * E*10, torch.zeros(samples_y, 2, device=device))

    # no fundamental changes
    forces_right = torch.ones(samples_y, device=device)
    forces_right[[0, -1]] = 0.5 * torch.ones(2, device=device)
    forces_right *= force_right * delta_y

    # Factor of 10 introduced to improve the compliance with this B.C.
    tractions_right = sig[-1, :, 0, 0] * delta_y
    tractions_right[[0, -1]] *= 0.5
    loss_right = 10*criterion(tractions_right, forces_right)

    # no changes
    sheartractions_right = sig[-1, :, 0, 1] * delta_y
    loss_right += criterion(sheartractions_right, torch.zeros(samples_y, device=device))
    tractions_lower = sig[1:-1, 0, 1, 1] * delta_x
    loss_lower = criterion(tractions_lower, torch.zeros(samples_x - 2, device=device))
    sheartractions_lower = sig[1:-1, 0, 1, 0] * delta_x
    loss_lower += criterion(sheartractions_lower, torch.zeros(samples_x - 2, device=device))
    tractions_upper = sig[1:-1, -1, 1, 1] * delta_x
    loss_upper = criterion(tractions_upper, torch.zeros(samples_x - 2, device=device))
    sheartractions_upper = sig[1:-1, -1, 1, 0] * delta_x
    loss_upper += criterion(sheartractions_upper, torch.zeros(samples_x - 2, device=device))

    print("loss_left: ", loss_left)
    print("loss_right: ", loss_right)

    bc_losses = loss_right #loss_left + loss_right + loss_upper + loss_lower
    return bc_losses

def losses(U, plot, data_driven):
    eps = geteps(sample_points, U)
    psi, sig = material_model(eps)

    # Cool plots to better understand what's going on
    if plot:
        plt.figure(figsize=(25, 25))
        plt.subplot(7, 3, 1)
        plt.gca().set_aspect('equal')
        plt.scatter((sample_points[:, 0] + U[:, 0]).detach().cpu().numpy(),
                    (sample_points[:, 1] + U[:, 1]).detach().cpu().numpy(),
                    c=(eps[:, :, 0, 0].detach().cpu().numpy()))
        plt.colorbar()
        plt.title("eps_x")
        plt.subplot(7, 3, 2)
        plt.gca().set_aspect('equal')
        plt.scatter((sample_points[:, 0] + U[:, 0]).detach().cpu().numpy(),
                    (sample_points[:, 1] + U[:, 1]).detach().cpu().numpy(),
                    c=(eps[:, :, 1, 1].detach().cpu().numpy()))
        plt.colorbar()
        plt.title("eps_y")
        plt.subplot(7, 3, 3)
        plt.gca().set_aspect('equal')
        plt.scatter((sample_points[:, 0] + U[:, 0]).detach().cpu().numpy(),
                    (sample_points[:, 1] + U[:, 1]).detach().cpu().numpy(),
                    c=(eps[:, :, 0, 1].detach().cpu().numpy()))
        plt.title("eps_xy")
        plt.colorbar()
        plt.subplot(7, 3, 4)
        plt.gca().set_aspect('equal')
        plt.scatter((sample_points[:, 0] + U[:, 0]).detach().cpu().numpy(),
                    (sample_points[:, 1] + U[:, 1]).detach().cpu().numpy(),
                    c=(psi[:, :].detach().cpu().numpy()))
        plt.colorbar()
        plt.title("psi")

        plt.subplot(7, 3, 7)
        plt.gca().set_aspect('equal')
        plt.scatter((sample_points[:, 0] + U[:, 0]).detach().cpu().numpy(),
                    (sample_points[:, 1] + U[:, 1]).detach().cpu().numpy(),
                    c=(U[:, 0].detach().cpu().numpy()))
        plt.title("U_x")
        plt.colorbar()
        plt.subplot(7, 3, 8)
        plt.gca().set_aspect('equal')
        plt.scatter((sample_points[:, 0] + U[:, 0]).detach().cpu().numpy(),
                    (sample_points[:, 1] + U[:, 1]).detach().cpu().numpy(),
                    c=(U[:, 1].detach().cpu().numpy()))
        plt.title("U_y")
        plt.colorbar()

        plt.tight_layout()


    U = U.reshape(samples_x, samples_y, 2)
    PSI = integratePsi(psi)
    T = integrateTractionEnergy(U, sig)
    bc_losses = boundaryLosses(U, sig)
    # The divergence loss result is not used anymore,
    # but calculated and printed for more insight into the behavior/ what's going on

    div_losses = calculateDivergenceLoss(sig)
    print("T-PSI: ", T - PSI)
    print("PSI: ", PSI)
    print("T: ", T)

    # New: calculate an _approximated_ analytical solution
    # (Actually, the lateral shrinkage is not linear!)
    # It can be used for a data-driven training of the NN --> see, if the NN architecture
    # is qualitatively capable of approximating the solution.
    # It also helped to find bugs if the NN was trained data-driven first and then physics-informed, afterwards.
    delta_l = force_right/(E*1/Lx)
    U_target = torch.zeros_like(sample_points)
    U_target[:, 0] = sample_points[:, 0]/Lx * delta_l
    U_target[:, 1] = -sample_points[:, 0]/Lx * nu * (sample_points[:, 1] / Ly - 0.5) * delta_l

    # Plot the _approximated solution_
    if plot:
        plt.subplot(7, 3, 10)
        plt.gca().set_aspect('equal')
        plt.scatter((sample_points[:, 0] + U_target[:, 0]).detach().cpu().numpy(),
                    (sample_points[:, 1] + U_target[:, 1]).detach().cpu().numpy(),
                    c='black')
        plt.title("deformed")

        eps = geteps(sample_points, U_target)

        plt.subplot(7, 3, 13)
        plt.gca().set_aspect('equal')
        plt.scatter((sample_points[:, 0] + U_target[:, 0]).detach().cpu().numpy(),
                    (sample_points[:, 1] + U_target[:, 1]).detach().cpu().numpy(),
                    c=(eps[:, :, 0, 0].detach().cpu().numpy()))
        plt.colorbar()
        plt.title("eps_x")

        plt.subplot(7, 3, 14)
        plt.gca().set_aspect('equal')
        plt.scatter((sample_points[:, 0] + U_target[:, 0]).detach().cpu().numpy(),
                    (sample_points[:, 1] + U_target[:, 1]).detach().cpu().numpy(),
                    c=(eps[:, :, 1, 1].detach().cpu().numpy()))
        plt.colorbar()
        plt.title("eps_y")

        plt.subplot(7, 3, 15)
        plt.gca().set_aspect('equal')
        plt.scatter((sample_points[:, 0] + U_target[:, 0]).detach().cpu().numpy(),
                    (sample_points[:, 1] + U_target[:, 1]).detach().cpu().numpy(),
                    c=(eps[:, :, 0, 1].detach().cpu().numpy()))
        plt.colorbar()
        plt.title("eps_xy")

        plt.show()

    # For data-driven training
    U_target = U_target.reshape(samples_x, samples_y, 2)
    criterion = nn.MSELoss(reduction="sum")

    if data_driven:
        print ("data_driven")
        return criterion(U, U_target)
    else:
        return torch.abs(T - PSI) + 1 * bc_losses  + 0 * div_losses



# This allows to choose another (random) initialization for the NN parameters
def weights_init(m):
    if isinstance(m, nn.Linear):
        #torch.nn.init.uniform_(m.weight, -1, 1)
        torch.nn.init.normal_(m.weight, mean=0.5/hidden_dim, std=10*1.0/hidden_dim)
        torch.nn.init.zeros_(m.bias)

# This is a custom activation function, that even has an own NN parameter (self.slope)
class myPow(nn.Module):
    def __init__(self):
        super().__init__()
        self.slope = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return torch.pow(x, 2) + self.slope * x

class MLNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLNet, self).__init__()
        self.hidden_dim = hidden_dim

        self.fcnn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            #nn.Tanh(),
            myPow(), #--> so easy is it to use a custom activation function
            # Which one is better? Try it out!
            nn.Linear(hidden_dim, output_dim, bias=True),
        )
        #self.fcnn1.apply(weights_init) --> comment in to apply the custom (random) intialization function

    def forward(self, x):
        #x_in = torch.cat((x[:, 0].unsqueeze(1)/Lx, x[:, 1].unsqueeze(1)/Ly - 0.5), dim=1) -> Normalize inputs, try it out!
        x_in = x
        out = force_right/(E*1/Lx) * self.fcnn1(x_in)
        #out = self.fcnn1(x)
        out[:, 0] *= x[:, 0]
        out[:, 1] *= x[:, 0] #--> In the literature a common output transformation, not proven to be necessary, here
        # In fact, that includes problem specific information and obstructs generalizability of the method
        # to a certain extent
        return out

# Should the training start data-driven with the _approximated_ solution as target values?
data_driven = False #True

def train():
    model = MLNet(input_dim, hidden_dim, output_dim)
    model.to(device)
    #optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01, max_iter=20)#, line_search_fn='strong_wolfe')
    # Adam has given much better results than LBFGS during the test runs for this sourcecode version.
    # lr=0.01 is not optimized, but this order of magnitude performed comparatively well.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 10000
    for epoch in range(epochs):
        model.train()
        #if (epoch + 1) % 3 == 0:
        #    global force_right
        #    if force_right < 0.05:
        #        force_right += 0.03
        if epoch > 1000:
            global data_driven
            data_driven = False # --> the following epochs will be a kind of transfer learning
        if (epoch) % 1000 == 0:
            U = model(sample_points)
            print("force_right: ", force_right)
            losses(U, True, True)
            plt.show()

        def closure():
            U = model(sample_points)
            global data_driven
            loss = losses(U, False, data_driven)
            print('Epoch %i/%i, Total Loss: %.64e' % (epoch+1, epochs, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            return loss
        loss = closure()
        optimizer.step()
        #optimizer.step(closure)
    return

train()