import torch

a = 10
x = torch.rand(10**7)*a

def f(x):
    return torch.sin(x)

print(torch.sum(f(x))*a**2/len(x))
print(torch.sum(f(x))*a/len(x))