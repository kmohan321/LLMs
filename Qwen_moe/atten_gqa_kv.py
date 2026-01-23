import torch
import torch.nn as nn
from common import RMSE_NORM, apply_rope

class GQA(nn.Module):
  def __init__(self,
               num_kv_heads: int,
               num_q_heads: int,
               hidden_dim: int,
               head_dim: int,
               qk_norm: bool = True,
               causal: bool = True,
               attention_bias: bool = False
               ):
    super().__init__()
    
    assert (hidden_dim % num_q_heads) == 0, "hidden_dim must be div by num_q_heads"
    
    self.qk_norm = qk_norm
    self.kv_heads = num_kv_heads
    self.q_heads = num_q_heads
    self.head_dim = head_dim
    self.num_groups = self.q_heads // self.kv_heads
    self.causal = causal
    
    self.wq = nn.Linear(hidden_dim, num_q_heads*self.head_dim, bias=attention_bias)
    self.wk = nn.Linear(hidden_dim, num_kv_heads*self.head_dim, bias=attention_bias)
    self.wv = nn.Linear(hidden_dim, num_kv_heads*self.head_dim, bias=attention_bias)
    self.wo = nn.Linear(num_q_heads*self.head_dim, hidden_dim, bias=attention_bias)
    
    self.scale = self.head_dim**-0.5
    
    self.q_norm = RMSE_NORM(self.head_dim)
    self.k_norm = RMSE_NORM(self.head_dim)
    
  def forward(self, x, freq, past_kv=None):
    b,s,d = x.shape
    q,k,v = self.wq(x), self.wk(x), self.wv(x)
    q = q.view(b, s, self.q_heads, self.head_dim)
    k = k.view(b, s, self.kv_heads, self.head_dim)
    v = v.view(b, s, self.kv_heads, self.head_dim)
    
    q,k,v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2) #(b,s,h,d) -> (b,h,s,d)
    
    if self.qk_norm:
      q = self.q_norm(q)
      k = self.k_norm(k)
     
    s_past = past_kv[0].shape[2] if past_kv is not None else 0
    q = apply_rope(q, freq[:, :, s_past :])
    k = apply_rope(k, freq[:, :, s_past :])
        
    if past_kv is not None:
      k = torch.cat([past_kv[0],k],dim=2)
      v = torch.cat([past_kv[1],v],dim=2)
    
    k_new, v_new = k, v
    
    Tq = q.shape[2]
    Tk = k.shape[2]
    
    #for gqa
    if self.num_groups > 1:
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)
    
    atten_scores = q @ k.transpose(2,3) * self.scale
    
    if self.causal:
      full_mask = torch.triu(torch.ones(Tk, Tk, dtype=torch.bool, device=x.device), diagonal=1)
      mask = full_mask[-Tq:, :]  # (Tq, Tk)
      atten_scores = atten_scores.masked_fill(mask, float('-inf'))
    
    atten_scores = torch.softmax(atten_scores,dim=-1)
    out = atten_scores @ v
    out = out.transpose(1,2).contiguous()
    return self.wo(out.view(b,s,-1)), k_new, v_new
    