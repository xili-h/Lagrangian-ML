import torch
from lagrangian import Net2, Net3, plot_net, DEVICE, get_psi, initial_wavepacket_3
import matplotlib.pyplot as plt
from Animation import Animation

# net = Net2()
#net.load_state_dict(torch.load("./bnet1_test1.pth"))
# net.to(DEVICE)

main_net = Net3(initial_wavepacket_3)

main_net.to(DEVICE)
# boundary_net = Net1()
# boundary_net.to(DEVICE)

#boundary_net.load_state_dict(torch.load("./bnet1_test1.pth"))
main_net.load_state_dict(torch.load("./mnet.pth"))

time_len = 5
# def boundary_and_main(x):
#     with torch.no_grad():
#         #boundary = boundary_net(x)
#         boundary = initial_wavepacket(x[:,1])[:, None]
#     main = main_net(x)
    
#     time_grid = x[:,0][:, None]
#     total = boundary*(1- time_grid/time_len)**2 + main*time_grid/time_len  #EQ2.6 of 2205.00593
#     return total

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
    # run_animation(boundary_and_main)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plot_net(main_net, ax)
    plt.show()


    # space = torch.linspace(-10, 10, 1000, device=DEVICE)
    # x_at_time_0 = torch.cat((torch.zeros_like(space[:,None]),space[:,None]), dim=-1)
    # psi = get_psi(boundary_and_main(x_at_time_0))
    # dx = 20/(1000-1)
    # psi2 = (torch.conj(psi)*psi).real
    # N = dx*torch.sum(psi2)
    # psi2 = (psi2/N).flatten()
    # psi = (psi/N).flatten()
    # plt.plot(space.cpu(), psi.imag.cpu()-initial_wavepacket(space).imag.cpu())
    # plt.show()
