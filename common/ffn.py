import torch
import torch.nn as nn

class FFN_SW(nn.Module):
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
  
class FFN(nn.Module):
  def __init__(self, hidden_dim, act, mlp_mutiplier, drop = 0.0):
    super().__init__()
    
    self.w1 = nn.Linear(hidden_dim,hidden_dim*mlp_mutiplier)
    self.act = act
    self.w2 = nn.Linear(mlp_mutiplier*hidden_dim,hidden_dim)
    self.dropout = nn.Dropout(drop)
    
  def forward(self,x):
    x = self.dropout(self.w1(x))
    x = self.act(x)
    x = self.w2(x)
    return x