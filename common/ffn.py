import torch
import torch.nn as nn

class FFN(nn.Module):
  def __init__(self,
               hidden_dim: int,
               intermediate_dim: int
               ):
    super().__init__()
    
    self.w1 = nn.Linear(hidden_dim, intermediate_dim,bias=False)
    self.w2 = nn.Linear(intermediate_dim, hidden_dim,bias=False)
    self.w3 = nn.Linear(hidden_dim, intermediate_dim,bias=False)
    self.act = nn.SiLU()
    
  def forward(self,x):
    gated_value = self.act(self.w1(x)) 
    x = gated_value * self.w3(x) #swiglu
    return self.w2(x)