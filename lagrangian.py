#%%
import torch
import numpy as np
import math
from random import random

#%%
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class Net(nn.Module):
    def __init__(self, space_period=20):
        super().__init__()
        self.time_fc = nn.Linear(1, 32)
        self.space_periodic = Periodic(1, 32, space_period,bias=False)
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
    def __init__(self, space_period=20):
        super().__init__()
        self.time_fc = nn.Linear(1, 32)
        self.space_periodic = Periodic(1, 32, space_period,bias=False)
        self.init_periodic_fc = nn.Linear(64, 32*32) 
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 240)
        self.fc2 = nn.Linear(240, 168)
        self.fc3 = nn.Linear(168, 20)
        self.fc4 = nn.Linear(20, 8)
        self.fc5 = nn.Linear(8, 2)

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
        x = F.tanh(self.fc4(x))
        x = self.fc5(x)
        return x
    

        
net = Net2()

if __name__ == '__main__':
    #check periodic
    with torch.no_grad():
        print(net(torch.tensor([[1.,2.],[1.,-18.],[1.,22.]])))

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

def complex_norm_loss(output, target):
    return (output - target).norm(dtype=torch.complex64)

def equation_motion_loss(psi, n_t, n_x, time_len, space_period):
    """Schrödinger equation with Monte Carlo integration"""
    psi = psi.reshape(n_t,n_x)

    time_int = torch.sum(psi)/n_t*time_len
    space_int2 = torch.sum(psi)/n_x*space_period**2

    return (1j*space_int2+0.5*time_int).norm(dtype=torch.complex64)

def get_space_time_grid(n_t,n_x,
            t_0=0, t_len=5,x_0=-10,space_period=20):
    space_lin = torch.linspace(x_0, x_0+space_period, n_x, device=DEVICE)
    time_lin = torch.linspace(t_0, t_0+t_len, n_t, device=DEVICE)
    time, space = torch.meshgrid(time_lin,space_lin,indexing='ij')
    x = torch.cat((time[:,:,None],space[:,:,None]),-1)
    x = torch.flatten(x, end_dim=-2)    
    return x, time_lin, space_lin

def get_rand_space_time_grid(n_t,n_x,
            t_0=0, t_len=5,x_0=-10,space_period=20):
    dt = t_len/(n_t-1)
    dx = space_period/(n_x-1)
    x_0 = x_0+random()*dx
    t_0 = t_0+random()*dt
    x, time_lin, space_lin = get_space_time_grid(n_t, n_x, t_0, t_len,x_0,space_period)
    return x, time_lin, space_lin, dt, dx

def get_rand_space_time_pair(n_t,n_x,
            t_0=0, t_len=5,x_0=-10,space_period=20):
    rand_time = t_0 + torch.rand(n_t, device=DEVICE)*t_len
    rand_space = x_0 + torch.rand(n_x, device=DEVICE)*space_period
    x = torch.cat((rand_time[:,None],rand_space[:,None]), dim=-1)
    return x, rand_time, rand_space

def get_psi(net_output):
    real, img = torch.split(net_output,1,-1)
    psi = real+img*1j #wave function
    return psi

def initial_wavepacket(space, x_0=-5, sigma=0.5, k0=-5):
    wave = (sigma*np.sqrt(np.pi))**(-0.5)*torch.exp(-(space-x_0)**2/(2*sigma**2) + (k0*space)*1j)
    return wave
def initial_wavepacket_2(space, x_0=0, sigma=0.5, k0=5):
    wave = (sigma*np.sqrt(np.pi))**(-0.5)*torch.exp( -(space-x_0)**2/(2*sigma**2))
    return wave*np.cos(k0*x) + wave*np.sin(k0*x)*1j

#%%

def plot_net(net: Net, ax, n_t: int=100, n_x:int =100,
            t_0=0, t_len=5,x_0=-10,space_period=20):
    space = torch.linspace(x_0, x_0+space_period, n_x, device=DEVICE)
    time = torch.linspace(t_0, t_0+t_len, n_t, device=DEVICE)
    time, space = torch.meshgrid(time,space,indexing='ij')
    x = torch.cat((time[:,:,None],space[:,:,None]),-1)
    x = torch.flatten(x, end_dim=-2)

    psi = get_psi(net(x))
    psi = psi.reshape(n_t,n_x)

    #dt = t_len/(n_t-1)
    dx = space_period/(n_x-1)
    psi2 = (torch.conj(psi)*psi).real
    N = dx*torch.sum(psi2,dim=1)[:,None]
    psi2 = psi2/N

    ax.clear()
    ax.plot_surface(time.cpu(),space.cpu(),psi2.cpu())
    plt.draw()
    plt.pause(0.1)

if __name__ == "__main__":
    optimizer = optim.Adam(net.parameters(), lr=0.000001)
    net.to(DEVICE)
    net.load_state_dict(torch.load("./lnet2_k_neg.pth"))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt.ion()
    plt.show(block=False)

    n_x=100
    n_t=100

    #x, time_lin, space_lin, dt, dx = get_rand_space_time_grid(n_t, n_x)
    #inital_condition = initial_wavepacket(space_lin)
    # for epoch in range(100):  # loop over the dataset multiple times

    #     x, rand_time, rand_space = get_rand_space_time_pair(n_t**2, n_x**2)
    #     inital_condition = initial_wavepacket(rand_space)
        
    #     # zero the parameter gradients
    #     optimizer.zero_grad()
    #     # forward + backward + optimize
    #     psi = get_psi(net(x))
    #     #psi = psi.reshape(n_t,n_x) #for grid 
    #     psi = psi.flatten()
    #     loss = complex_norm_loss(psi, inital_condition)

    #     loss.backward()
    #     optimizer.step()

    #     # print statistics
    #     running_loss = loss.item()
    #     if epoch % 10 == 9:
    #         print(f'[{epoch + 1}] loss: {running_loss}')
    #     if epoch % 10 == 9:
    #         with torch.no_grad():
    #             net.to("cpu")
    #             plot_net(net,ax)
    #             net.to(DEVICE)

    # for epoch in range(10**4):  # loop over the dataset multiple times
    #     x, rand_time, rand_space = get_rand_space_time_pair(n_t**2, n_x**2)
        
    #     # zero the parameter gradients
    #     optimizer.zero_grad()

    #     # forward + backward + optimize
        
    #     psi = get_psi(net(x))
    #     psi = psi.reshape(n_t,n_x) #for grid 
    #     M_loss = equation_motion_loss(psi, n_t, n_x, time_len=3, space_period=20)

    #     inital_bound = torch.cat((torch.zeros_like(rand_space[:,None]),rand_space[:,None]),dim=-1)
    #     psi = get_psi(net(inital_bound)).flatten()
    #     inital_condition = initial_wavepacket(rand_space)
    #     inital_loss = complex_norm_loss(psi,inital_condition)

    #     loss = inital_loss*M_loss

    #     loss.backward()
    #     optimizer.step()

    #     # print statistics
    #     running_loss = loss.item()
    #     if epoch % 50 == 49:
    #         print(f'[{epoch + 1}] loss: {running_loss}')
    #     if epoch % 50 == 49:
    #         with torch.no_grad():
    #             net.to("cpu")
    #             plot_net(net,ax)
    #             net.to("cuda")
    
    for epoch in range(2*10**4):  # loop over the dataset multiple times
        x, time_lin, space_lin, dt, dx = get_rand_space_time_grid(n_t, n_x)
        inital_condition = initial_wavepacket(space_lin)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        
        psi = get_psi(net(x))
        psi = psi.reshape(n_t,n_x) #for grid 
        L_loss = lagrangian_loss(psi,dt,dx)
        inital_loss = complex_norm_loss(psi[0],inital_condition)
        loss = L_loss+inital_loss*10**3

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss = loss.item()
        if epoch % 100 == 99:
            print(f'[{epoch + 1}] totol_loss: {running_loss}, inital_loss:{inital_loss}')
        if epoch % 1000 == 999:
            with torch.no_grad():
                plot_net(net,ax)

    print('Finished Training')
    PATH = './lnet.pth'
    torch.save(net.state_dict(), PATH)

#%%


