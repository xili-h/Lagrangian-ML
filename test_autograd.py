import torch

x =  torch.tensor([1., 2., 3., 4.])
x.requires_grad_(True)
y = x**2 + 1j*x**3
#y = torch.stack((y,y))

#v = torch.tensor([1,1,0,0])
v= torch.ones_like(y) +torch.ones_like(y)*1j

print(torch.autograd.grad(y,x,v))