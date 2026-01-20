import torch
import torch.nn as nn
from common import apply_rope

class GQA(nn.Module):
  def __init__(self,
               hidden_dims :int,
               num_heads_q: int,
               num_heads_kv :int
               ):
    super().__init__()
    
    self.num_heads_q = num_heads_q
    self.num_heads_kv = num_heads_kv
    
    assert hidden_dims % self.num_heads_q==0, 'hidden_dim must be divisible by num_heads'
    assert self.num_heads_q % self.num_heads_kv == 0, "num_heads_q must be divisible by num_heads_kv"
    
    self.head_dim = hidden_dims//self.num_heads_q
    self.groups = self.num_heads_q//self.num_heads_kv
    
    self.wq = nn.Linear(hidden_dims, num_heads_q*self.head_dim, bias=False)
    self.wk = nn.Linear(hidden_dims, num_heads_kv*self.head_dim, bias=False) 
    self.wv = nn.Linear(hidden_dims, num_heads_kv*self.head_dim, bias=False)
    self.wo = nn.Linear(self.num_heads_q*self.head_dim,hidden_dims,bias=False)
  
  def forward(self, x: torch.Tensor, freq: torch.Tensor, mask: torch.Tensor, is_causal = True):
    
    b, s, d = x.shape
    query = self.wq(x).view(b,s,self.num_heads_q,self.head_dim).transpose(1,2)
    key = self.wk(x).view(b,s,self.num_heads_kv,self.head_dim).transpose(1,2)
    value = self.wv(x).view(b,s,self.num_heads_kv,self.head_dim).transpose(1,2)
    
    #(b, h, s, d)
    rotated_query = apply_rope(query, freq)
    rotated_key = apply_rope(key, freq)
    
    if self.groups > 1:
      rotated_key = rotated_key.repeat_interleave(self.groups, dim=1)
      value = value.repeat_interleave(self.groups, dim=1)
      
    attention_score = rotated_query @ rotated_key.transpose(2,3)
    if is_causal:
      attention_score = torch.masked_fill(attention_score, mask = mask[:s,:s]==0, value = -torch.inf)
    attention_weights = torch.softmax(attention_score * (self.head_dim**-0.5),dim=-1)
    
    out = attention_weights @ value
    out = out.transpose(1,2)
    out = out.contiguous().view(b, s, self.num_heads_q * self.head_dim)
    
    return self.wo(out)
  
