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
  
  def forward(self, x: torch.Tensor, freq: torch.Tensor, past_kv, is_causal = True):
    
    b, s, d = x.shape
    q = self.wq(x).view(b,s,self.num_heads_q,self.head_dim).transpose(1,2)
    k = self.wk(x).view(b,s,self.num_heads_kv,self.head_dim).transpose(1,2)
    v = self.wv(x).view(b,s,self.num_heads_kv,self.head_dim).transpose(1,2)
    
    #(b, h, s, d)
    seq_len = past_kv[0].shape[2] if past_kv is not None else 0
    q = apply_rope(q, freq[:, :, seq_len:])
    k = apply_rope(k, freq[:, :, seq_len:])
    
    if past_kv is not None:
      k = torch.cat([past_kv[0], k], dim = 2)
      v = torch.cat([past_kv[1], v], dim = 2)
      
    k_new, v_new = k, v
    
    if self.groups > 1:
      k = k.repeat_interleave(self.groups, dim=1)
      v = v.repeat_interleave(self.groups, dim=1)
    
    Tk = k.shape[2]
    Tq = q.shape[2]
      
    attention_score = q @ k.transpose(2,3)
    if is_causal:
      full_mask = torch.tril(torch.ones(Tk, Tk, dtype=torch.bool, device=x.device))
      full_mask = full_mask[-Tq:, :]
      attention_score = torch.masked_fill(attention_score, mask = (full_mask==0), value = -torch.inf)
    attention_weights = torch.softmax(attention_score * (self.head_dim**-0.5),dim=-1)
    
    out = attention_weights @ v
    out = out.transpose(1,2)
    out = out.contiguous().view(b, s, self.num_heads_q * self.head_dim)
    
    return self.wo(out), k_new, v_new
  
