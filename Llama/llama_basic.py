import torch
import torch.nn as nn
from SelfAttention_GQ import SELF_ATTENTION

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
               ffn_multiplier:int
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
    
    self.atten = SELF_ATTENTION(config["hidden_dims"], config["num_heads_q"], config["num_heads_kv"], config["seq_length"])
    self.norm1 = RMSE_NORM(config["hidden_dims"], config["eps"])
    self.norm2 = RMSE_NORM(config["hidden_dims"], config["eps"])
    self.ffn = FFN(config["hidden_dims"], config["ffn_multiplier"])
    
  def forward(self,x):
    x = x + self.atten(self.norm1(x))
    x = self.ffn(self.norm2(x))
    return x
  
class LLama_Basic(nn.Module):
  def __init__(self,config):
    super().__init__()    
    
    self.blocks = nn.ModuleList([Attention_Block(config) for _ in range(config["num_blocks"])])
    self.embedding = nn.Embedding(config["vocab_size"],config["hidden_dims"])
    self.final_layer = nn.Linear(config["hidden_dims"],config["vocab_size"],bias=False)
    self.rmse_norm = RMSE_NORM(config["hidden_dims"],config["eps"])
    
  def forward(self,tokens):
    x = self.embedding(tokens)
    
    for block in self.blocks:
      x = block(x)
    
    x = self.rmse_norm(x)
    return self.final_layer(x)
      
# config = {
#     "model": {
#         "num_blocks": 2,
#         "hidden_dims": 256,
#         "num_heads_q": 8,
#         "num_heads_kv": 4,
#         "seq_length": 4096,
#         "ffn_multiplier": 4,
#         "vocab_size": 52000,
#         "eps": 1e-5
#     }
# }

    
    
    
    
  