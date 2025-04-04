import torch
import torch.nn as nn
from Gqa_sa import GQA_SA
from utils.rope import Frequencies

class RMSE_NORM(nn.Module):
  def __init__(self,
               hidden_dim:int,
               eps:float = 1e-5
               ):
    super().__init__()
    self.eps = eps
    self.shift = nn.Parameter(torch.ones(hidden_dim))

  def forward(self,x):
    x = x * torch.rsqrt((x**2).mean(dim=-1, keepdim=True) + self.eps)
    return x  * self.shift

class FFN(nn.Module):
  def __init__(self,
               hidden_dim:int,
               ffn_multiplier: float
               ):
    super().__init__()
    
    self.w1 = nn.Linear(hidden_dim,ffn_multiplier*hidden_dim,bias=False)
    self.w2 = nn.Linear(ffn_multiplier*hidden_dim,hidden_dim,bias=False)
    self.w3 = nn.Linear(hidden_dim,ffn_multiplier*hidden_dim,bias=False)
    self.act = nn.SiLU()
    
  def forward(self,x):
    x1 = self.act(self.w1(x)) 
    gated_value = self.w3(x)
    x = x1 * gated_value #swiglu
    return self.w2(x)
  
class Attention_Block(nn.Module):
  def __init__(self,config):
    super().__init__()
    
    self.atten = GQA_SA(config["num_heads_q"],  config["num_heads_kv"], config["hidden_dims"])
    self.norm1 = RMSE_NORM(config["hidden_dims"], config["eps"])
    self.norm2 = RMSE_NORM(config["hidden_dims"], config["eps"])
    self.ffn = FFN(config["hidden_dims"], config["ffn_multiplier"])
    
  def forward(self,x,freq,mask):
    x = x + self.atten(self.norm1(x),freq,mask)
    x = self.ffn(self.norm2(x))
    return x
  
class Mistral(nn.Module):
  def __init__(self,config):
    super().__init__()    
    
    self.blocks = nn.ModuleList([Attention_Block(config) for _ in range(config["num_blocks"])])
    self.embedding = nn.Embedding(config["vocab_size"],config["hidden_dims"])
    self.final_layer = nn.Linear(config["hidden_dims"],config["vocab_size"],bias=False)
    self.rmse_norm = RMSE_NORM(config["hidden_dims"],config["eps"])
    
    self.rotatory_freq = Frequencies(config["seq_length"],config["head_dim"])
    self.window_size = config["window_size"]
    
  def forward(self,tokens):
    x = self.embedding(tokens)
    _,s,_ = x.shape
    
    complex_frequncies = self.rotatory_freq.complex_freq
    i = torch.arange(0,s).view(s,1)
    j = torch.arange(0,s).view(1,s)
    mask = (i>=j) & (j >= i - self.window_size + 1)
    for block in self.blocks:
      x = block(x,complex_frequncies,mask)
    
    x = self.rmse_norm(x)
    return self.final_layer(x)
      
config = {
    "model": {
        "num_blocks": 2,
        "hidden_dims": 256,
        "num_heads_q": 8,
        "num_heads_kv": 4,
        "seq_length": 4096,
        "ffn_multiplier": 4,
        "vocab_size": 52000,
        "eps": 1e-5,
        "head_dim" : 32,
        "window_size" : 4
    }
}

# mis = Mistral(config["model"])
# x = torch.randint(0,config["model"]["vocab_size"],size=(2,config["model"]["seq_length"]))
# print(mis(x).shape)
# print(mis)

    
    
    
    
  
    
    
    