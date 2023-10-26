import torch
from lagrangian import Net2, get_space_time_grid, DEVICE,get_psi
import matplotlib.pyplot as plt
from Animation import Animation

net = Net2()
net.load_state_dict(torch.load("./lnet2_k_neg2_larger_inital.pth"))
net.to(DEVICE)


# n_x=n_t=100
# t_f=3
# x_0=-10
# space_period=20
# space = torch.linspace(x_0, x_0+space_period, n_x)
# x, dt, dx = get_space_time_grid(n_t, n_x)

def plot_net(net, ax, n_t: int=100, n_x:int =100,
            t_0=0, t_len=5,x_0=-10,space_period=20):
    space = torch.linspace(x_0, x_0+space_period, n_x, device=DEVICE)
    time = torch.linspace(t_0, t_0+t_len, n_t, device=DEVICE)
    time, space = torch.meshgrid(time,space,indexing='ij')
    x = torch.cat((time[:,:,None],space[:,:,None]),-1)
    x = torch.flatten(x, end_dim=-2)

    psi = get_psi(net(x))
    psi = psi.reshape(n_t,n_x)

    psi2 = (torch.conj(psi)*psi).real
    psi2 = psi2/torch.sum(psi2,dim=1)[:,None]

    psi = psi.real/torch.sum(psi2,dim=1)[:,None]

    ax.clear()
    ax.plot_surface(time.cpu(),space.cpu(),psi.cpu())
    plt.draw()
    plt.pause(0.1)

def run_animation(net, n_t: int=100, n_x:int =100,
            t_0=0, t_len=5,x_0=-10,space_period=20):
    space_len = torch.linspace(x_0, x_0+space_period, n_x, device=DEVICE)
    time_len = torch.linspace(t_0, t_0+t_len, n_t, device=DEVICE)

    time, space = torch.meshgrid(time_len,space_len,indexing='ij')
    x = torch.cat((time[:,:,None],space[:,:,None]),-1)
    x = torch.flatten(x, end_dim=-2)

    psi = get_psi(net(x))
    psi = psi.reshape(n_t,n_x)

    dt = t_len/(n_t-1)
    dx = space_period/(n_x-1)
    psi2 = (torch.conj(psi)*psi).real
    N = dx*torch.sum(psi2,dim=1)[:,None]
    psi2 = psi2/N

    anim = Animation(time_len.cpu(), space_len.cpu(), psi2.cpu())
    anim.run_animtion(-50)

with torch.no_grad():
    run_animation(net)