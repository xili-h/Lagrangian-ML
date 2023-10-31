#%%
import torch
import numpy as np
import math
from random import random
import matplotlib.pyplot as plt

#%%
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
import torchvision
import deepxde as dde

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_t = 500
N_x = 500
T_0 = 0
TIME_LENGTH = 5
X_0 =-10
SPACE_PERIOD = 20
D_T = TIME_LENGTH/(N_t-1)
D_X = SPACE_PERIOD/(N_x-1)

class Inital_Condition(nn.Module):
    #Maybe 2004.06490
    pass

class Periodic(nn.Module):
    """Periodic Conditions (2007.07442) modify from nn.Linear

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    Attributes:
        weight_A: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        weight_phi: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    omega: Tensor
    weight_A: Tensor
    weight_phi: Tensor
    def __init__(self, in_features: int, node: int, period: Tensor, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = node*in_features
        self.omega = 2*math.pi/period
        self.weight_A = Parameter(torch.empty((self.out_features, in_features), **factory_kwargs))
        self.weight_phi = Parameter(torch.empty((self.out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(self.out_features, in_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_phi, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_A)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        x = self.weight_A[None,:]*torch.cos(self.omega*input[:,None]+self.weight_phi[None,:])
        if self.bias is not None:
            return x+self.bias[None,:]
        else:
            return x
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

class PeriodicResNet(nn.Module):
    def __init__(self, inital_condition=None):
        super().__init__()
        self.time_fc = nn.Linear(1, 32)
        self.space_periodic = Periodic(1, 32, SPACE_PERIOD)
        self.space_time_fc = nn.Linear(64, 32*32*3) 
        self.resnet = torchvision.models.resnet18()
        self.fc1 = nn.Linear(1000, 400)
        self.fc2 = nn.Linear(400, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 10)
        self.fc5 = nn.Linear(10, 2)
        self.inital_condition = inital_condition 

    def forward(self, x): #x = [time, space]
        time, space = torch.split(x,1,-1)
        time_x = self.time_fc(time)
        space_x = torch.flatten(self.space_periodic(space), 1)
        x = torch.cat((time_x, space_x),1) #x[0] is time, x[1] is space
        x = self.space_time_fc(x)
        x = x.reshape(-1,3,32,32)
        x = self.resnet(x)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        x = self.fc5(x)
        if self.inital_condition is not None:
            # frac = torch.minimum(torch.tensor([1], device=DEVICE),time/D_T)
            # x= self.inital_condition(space)*(1-frac) +  frac*x
            x= self.inital_condition(space) +  time/D_T*x
        return x

class Net1(nn.Module):
    def __init__(self, space_period=20):
        super().__init__()
        self.time_fc = nn.Linear(1, 32)
        self.space_periodic = Periodic(1, 32, space_period)
        self.init_periodic_fc = nn.Linear(64, 32*32) 
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 2)

    def forward(self, x): #x = [time, space]
        time, space = torch.split(x,1,-1)
        time = self.time_fc(time)
        space = torch.flatten(self.space_periodic(space), 1)
        x = torch.cat((time, space),1) #x[0] is time, x[1] is space
        x = self.init_periodic_fc(x)
        x = x.reshape(-1,1,32,32)
        x = self.pool(F.tanh(self.conv1(x)))
        x = self.pool(F.tanh(self.conv2(x)))
        x = torch.flatten(x,1) # flatten all dimensions except batch
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

class Net2(nn.Module):
    def __init__(self, inital_condition=None):
        super().__init__()
        self.time_fc = nn.Linear(1, 32)
        self.space_periodic = Periodic(1, 32, SPACE_PERIOD)
        self.init_periodic_fc = nn.Linear(64, 32*32) 
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 240)
        self.fc2 = nn.Linear(240, 168)
        self.fc3 = nn.Linear(168, 20)
        self.fc4 = nn.Linear(20, 8)
        self.fc5 = nn.Linear(8, 2)
        self.inital_condition = inital_condition

    def forward(self, x): #x = [time, space]
        time, space = torch.split(x,1,-1)
        time_x = self.time_fc(time)
        space_x = torch.flatten(self.space_periodic(space), 1)
        x = torch.cat((time_x, space_x),1) #x[0] is time, x[1] is space
        x = self.init_periodic_fc(x)
        x = x.reshape(-1,1,32,32)
        x = self.pool(F.tanh(self.conv1(x)))
        x = self.pool(F.tanh(self.conv2(x)))
        x = torch.flatten(x,1) # flatten all dimensions except batch
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        x = self.fc5(x)
        if self.inital_condition is not None:
            # frac = torch.minimum(torch.tensor([1], device=DEVICE),time/D_T)
            # x= self.inital_condition(space)*(1-frac) +  frac*x
            x= self.inital_condition(space) +  time/D_T*x
        return x
    
class Net3(nn.Module):
    def __init__(self, inital_condition=None):
        super().__init__()
        self.time_fc = nn.Linear(1, 32)
        self.space_periodic = Periodic(1, 32, SPACE_PERIOD)
        self.init_periodic_fc = nn.Linear(64, 32*32) 
        self.conv1 = nn.Conv2d(1, 15, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(15, 40, 5)
        self.fc1 = nn.Linear(40 * 5 * 5, 240)
        self.fc2 = nn.Linear(240, 168)
        self.fc3 = nn.Linear(168, 90)
        self.fc4 = nn.Linear(90, 20)
        self.fc5 = nn.Linear(20, 2)
        self.inital_condition = inital_condition

    def forward(self, x): #x = [time, space]
        time, space = torch.split(x,1,-1)
        time_x = self.time_fc(time)
        space_x = torch.flatten(self.space_periodic(space), 1)
        x = torch.cat((time_x, space_x),1) #x[0] is time, x[1] is space
        x = self.init_periodic_fc(x)
        x = x.reshape(-1,1,32,32)
        x = self.pool(F.tanh(self.conv1(x)))
        x = self.pool(F.tanh(self.conv2(x)))
        x = torch.flatten(x,1) # flatten all dimensions except batch
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        x = self.fc5(x)
        if self.inital_condition is not None:
            # frac = torch.minimum(torch.tensor([1], device=DEVICE),time/D_T)
            # x= self.inital_condition(space)*(1-frac) +  frac*x
            x= self.inital_condition(space) +  time/D_T*x
        return x
#%%

def time_D(psi, dt):
    time_pos = psi[1:, :-1]
    time_neg = torch.cat((psi[-2, :-1][None,:],psi[:-2, :-1]),dim=0)
    return (time_pos-time_neg)/(2*dt) #time[0]=time[-1], so time_D[-1] can be ignore, same for space

def space_D(psi, dx):
    space_pos = psi[:-1, 1:]
    space_neg = torch.cat((psi[:-1, -2][:,None],psi[:-1, :-2]),dim=1)
    return (space_pos-space_neg)/(2*dx) #time[0]=time[-1], so time_D[-1] can be ignore, same for space

def space_D2(psi, dx):
    space_pos = psi[:-1, 1:]
    space_neg = torch.cat((psi[:-1, -2][:,None],psi[:-1, :-2]),dim=1)
    space_mid = psi[:-1,:-1]
    return (space_pos-2*space_mid+space_neg)/dx**2

def lagrangian_loss(psi,dt,dx): #Schrödinger field
    #space = x_0+torch.rand(n_x)*space_period
    psi_star = torch.conj(psi)
    N = torch.conj(psi[:-1,:-1])*psi[:-1,:-1] #psi_star * psi
    L = 0.5j*( psi_star[:-1,:-1]*time_D(psi,dt) - psi[:-1,:-1]*time_D(psi_star,dt) ) \
        - 0.5*space_D(psi,dx)*space_D(psi_star,dx)
    
    #N = torch.sum(N, dim=1) #integrate over space
    #L = torch.sum(L, dim=1) # integrate over space

    #use mean of time space mean
    L = L.mean().real
    N = N.mean().real
    return L/N

def equation_motion_net_loss(net,x):
    net_out = net(x)
    real, imag = torch.split(net_out,1,-1)

    dreal_time = dde.grad.jacobian(real,x,i=0)[:,0]
    dimag_time = dde.grad.jacobian(imag,x,i=0)[:,0]

    dreal_space2 = dde.grad.jacobian(real,x,i=1)[:,1]
    dimag_space2 = dde.grad.jacobian(imag,x,i=1)[:,1]

    real_part = 0.5*dreal_space2 - dimag_time
    imag_part = 0.5*dimag_space2 + dreal_time

    # dreal_x = torch.autograd.grad(real,x,torch.ones_like(real), create_graph=True, retain_graph=True)[0]
    # dreal_xx = torch.autograd.grad(dreal_x,x,torch.ones_like(dreal_x), retain_graph=True)[0]

    # dimag_x = torch.autograd.grad(imag,x,torch.ones_like(imag), create_graph=True, retain_graph=True)[0]
    # dimag_xx = torch.autograd.grad(dimag_x,x,torch.ones_like(dimag_x), retain_graph=True)[0]

    # real_part = 0.5*dreal_xx[:,1] - dimag_x[:,0]
    # imag_part = 0.5*dimag_xx[:,1] + dreal_x[:,0]

    equation =  real_part + imag_part*1j
    return equation.norm(dtype=torch.complex64)

def complex_norm_loss(output, target):
    return (output - target).norm(dtype=torch.complex64)

def equation_motion_loss(psi,dt,dx):
    """Schrödinger equation"""
    equation = 1j*time_D(psi, dt)+0.5*space_D2(psi, dx)
    return (equation).norm(dtype=torch.complex64)

#%%

def get_psi(net_output):
    real, img = torch.split(net_output,1,-1)
    psi = real+img*1j #wave function
    return psi

def initial_wavepacket(space, x_0=-5, sigma=0.5, k0=5):
    wave = (sigma*math.sqrt(math.pi))**(-0.5)*torch.exp(-(space-x_0)**2/(2*sigma**2) + (k0*space)*1j)
    return wave

def initial_wavepacket_2(x, x_0=-5, sigma=0.5, k0=5):
    wave = (sigma*math.sqrt(math.pi))**(-0.5)*torch.exp( -(x-x_0)**2/(2*sigma**2))
    return  wave*torch.cos(k0*x), wave*torch.sin(k0*x)

def initial_wavepacket_3(x, x_0=-5, sigma=0.5, k0=5):
    wave = (sigma*math.sqrt(math.pi))**(-0.5)*torch.exp( -(x-x_0)**2/(2*sigma**2))
    return  torch.cat((wave*torch.cos(k0*x), wave*torch.sin(k0*x)), dim=-1)
#%%


def get_space_time_grid(t_0=T_0, x_0=X_0):
    space_lin = torch.linspace(x_0, x_0+SPACE_PERIOD, N_x, device=DEVICE)
    time_lin = torch.linspace(t_0, t_0+TIME_LENGTH, N_t, device=DEVICE)
    time, space = torch.meshgrid(time_lin,space_lin,indexing='ij')
    x = torch.cat((time[:,:,None],space[:,:,None]),-1)
    x = torch.flatten(x, end_dim=-2)    
    return x, time_lin, space_lin

def get_rand_space_time_pair():
    if N_t == N_x:
        rand_time = T_0 + torch.rand(N_t, device=DEVICE)*TIME_LENGTH
        rand_space = X_0 + torch.rand(N_x, device=DEVICE)*SPACE_PERIOD
        x = torch.cat((rand_time[:,None],rand_space[:,None]), dim=-1)
        return x, rand_time, rand_space
    else:
        raise RuntimeError("N_t not equal N_x")

def get_rand_space_time_grid():
    x_0 = X_0+random()*D_X
    t_0 = T_0+random()*D_T
    x, time_lin, space_lin = get_space_time_grid(t_0,x_0)
    return x, time_lin, space_lin

def get_rand_space():
    rand_space = X_0 + torch.rand(N_x, device=DEVICE)*SPACE_PERIOD
    return rand_space


#%%

def plot_net(net, ax):
    space = torch.linspace(X_0, X_0+SPACE_PERIOD, N_x, device=DEVICE)
    time = torch.linspace(T_0, T_0+TIME_LENGTH, N_t, device=DEVICE)
    time, space = torch.meshgrid(time,space,indexing='ij')
    x = torch.cat((time[:,:,None],space[:,:,None]),-1)
    x = torch.flatten(x, end_dim=-2)

    psi = get_psi(net(x))
    psi = psi.reshape(N_t,N_x)

    #dt = t_len/(n_t-1)
    #dx = space_period/(n_x-1)
    psi2 = (torch.conj(psi)*psi).real
    #N = D_X*torch.sum(psi2,dim=1)[:,None]
    psi2 = psi2 #/N

    ax.clear()
    ax.plot_surface(time.cpu(),space.cpu(),psi2.cpu())
    plt.draw()
    plt.pause(0.1)



if __name__ == "__main__":
    main_net = Net2(initial_wavepacket_3)
    main_net.to(DEVICE)

    # boundary_net = PeriodicResNet()
    # boundary_net.to(DEVICE)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(projection='3d')
    plt.ion()
    plt.show(block=False)

    # for lr in range(3,10):
    #     boundary_optimizer = optim.Adam(boundary_net.parameters(), lr=10**-lr)
    #     for epoch in range(lr*10**3):  # loop over the dataset multiple times
    #         rand_space = get_rand_space()
    #         inital_condition = initial_wavepacket(rand_space)
    #         x_at_T_0 = torch.cat((torch.zeros_like(rand_space[:,None]),rand_space[:,None]), dim=-1)
            
    #         # zero the parameter gradients
    #         boundary_optimizer.zero_grad()
    #         # forward + backward + optimize
    #         psi = get_psi(boundary_net(x_at_T_0))
    #         psi = psi.flatten()
    #         loss = complex_norm_loss(psi, inital_condition)

    #         loss.backward()
    #         boundary_optimizer.step()

    #         # print statistics
    #         running_loss = loss.item()
    #         if epoch % 100 == 99:
    #             print(f'[{epoch + 1}] loss: {running_loss}')
    #         if epoch % 500 == 499:
    #             with torch.no_grad():
    #                 plot_net(boundary_net,ax)
    # print('Finished Training')
    # PATH = './bnet.pth'
    # torch.save(boundary_net.state_dict(), PATH)

    # boundary_net.load_state_dict(torch.load("./bresnet_test1.pth"))

    #main_net.load_state_dict(torch.load("./mnet.pth"))
    main_optimizer = optim.Adam(main_net.parameters(), lr=0.001)
    #main_optimizer = optim.Adagrad(main_net.parameters())
    # main_optimizer = optim.LBFGS(main_net.parameters())
    
    # x, time_lin, space_lin = get_space_time_grid(n_t, n_x, t_len=time_len, space_period=space_period)
    # dt = time_len/(n_t-1)
    # dx = space_period/(n_x-1)

    # for epoch in range(10**2):
    #     main_optimizer.zero_grad()
    #     x, time_lin, space_lin = get_rand_space_time_grid()
    #     loss = torch.abs(main_net(x)).max()
    #     loss.backward()
    #     main_optimizer.step()

    #     # print statistics
    #     running_loss = loss.item()
    #     if epoch % 10 == 0:
    #         print(f'[{epoch + 1}] totol_loss: {running_loss}')
    #     if epoch % 20 == 0:
    #         with torch.no_grad():
    #             plot_net(main_net,ax)

    # def boundary_and_main(x):
    #     boundary = initial_wavepacket(x[:,1])[:, None]
    #     boundary = torch.cat((boundary.real, boundary.imag), dim=-1)
    #     main = main_net(x)
    #     #main = main*boundary.sum()/(main.sum())*2
        
    #     time_grid = x[:,0][:, None]
    #     total = boundary + main*time_grid/TIME_LENGTH  #EQ2.6 of 2205.00593
    #     return total

    for epoch in range(5*10**5): 
        #x, time_lin, space_lin = get_space_time_grid()
        x, time_lin, space_lin  = get_rand_space_time_pair()
        x.requires_grad_(True)
        
        # zero the parameter gradients
        #main_optimizer.zero_grad()

        # forward + backward + optimize
        #boundary = boundary_net(x)
        
        #psi = get_psi(main_net(x))
        #psi = psi.reshape(N_t,N_x) #for grid
        
        #loss = lagrangian_loss(psi,D_T,D_X)
        #loss = equation_motion_loss(psi,D_T,D_X)
        loss = equation_motion_net_loss(main_net,x)

        loss.backward()
        main_optimizer.step()

        # print statistics
        running_loss = loss.item()
        if epoch % 100 == 0:
            print(f'[{epoch + 1}] totol_loss: {running_loss}')
        if epoch % 200 == 0:
            with torch.no_grad():
                plot_net(main_net,ax)
                #plot_net(main_net,ax2)
        if epoch % 10**4 == 10**4-1:
            PATH = f'./mnet_epoch_{epoch}.pth'
            torch.save(main_net.state_dict(), PATH)

    print('Finished Training')
    PATH = './mnet.pth'
    torch.save(main_net.state_dict(), PATH)

#%%


