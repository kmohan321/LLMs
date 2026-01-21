import torch
import torch.nn as nn

class RMSE_NORM(nn.Module):
  def __init__(self,
               hidden_dim:int,
               eps:float = 1e-5
               ):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(hidden_dim))

  def forward(self, x):
    x = x * torch.rsqrt((x**2).mean(dim=-1, keepdim=True) + self.eps)
    return x  * self.weight
  

class LAYER_NORM(nn.Module):
  def __init__(self, hidden_dim, eps):
    super().__init__()
    
    self.eps = eps
    self.bias = nn.Parameter(torch.zeros(hidden_dim))
    self.weight = nn.Parameter(torch.ones(hidden_dim))
    
  def forward(self,x):
    mean = torch.mean(x,-1,keepdim=True)
    var = torch.var(x,-1,keepdim=True,unbiased=False)
    
    x = (x-mean)/torch.sqrt((var + self.eps))
    return x * self.weight + self.bias

