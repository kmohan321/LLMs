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
  


