import torch
import torch.nn as nn
from common import apply_rope

class GQA_SA(nn.Module):
  def __init__(self,
               q_heads:int,
               kv_heads:int,
               d :int
               ):
    super().__init__()
    
    assert d % q_heads == 0 , "hidden_dim must be divisible by query heads"
    assert q_heads % kv_heads ==0, "query heads must be divisible by kv heads"
    self.d_head = d // q_heads;
    self.q_heads = q_heads
    self.kv_heads = kv_heads
    self.wq = nn.Linear(d, self.d_head * q_heads)
    self.wk = nn.Linear(d, self.d_head * kv_heads)
    self.wv = nn.Linear(d, self.d_head * kv_heads)
    self.wo = nn.Linear(self.d_head * q_heads , d)
    
    self.query_groups = self.q_heads // self.kv_heads
    self.scale = self.d_head ** -0.5
    
  def forward(self, x: torch.Tensor, rope_freq : torch.Tensor, mask: torch.Tensor | None = None):

      b,s,d = x.shape
      q = self.wq(x).view(b, s, self.q_heads, self.d_head)
      k = self.wk(x).view(b, s, self.kv_heads, self.d_head)
      v = self.wv(x).view(b, s, self.kv_heads, self.d_head)
      
      q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
      q,k = apply_rope(q, rope_freq), apply_rope(k, rope_freq)
      
      if self.query_groups > 0:
        k = k.repeat_interleave(self.query_groups, dim = 1)
        v = v.repeat_interleave(self.query_groups, dim = 1)
        
      attn_scores = q @ k.transpose(2,3)
      if mask is not None:
        attn_scores = attn_scores + mask
      attn_scores = torch.softmax((attn_scores * self.scale), dim=-1)
      
      out = (attn_scores @ v).transpose(1,2).contiguous().view(b,s,d)
      return self.wo(out)
            