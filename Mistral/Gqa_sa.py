import torch
import torch.nn as nn
from utils import rope_apply

class GQA_SA(nn.Module):
  def __init__(self,
               q_heads:int,
               kv_heads:int,
               d :int,
               w : int
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
    
  def forward(self, x: torch.Tensor, rope_freq : torch.Tensor = None , mask: torch.Tensor = None):

      b,s,d = x.shape
      q = self.wq(x).view(b, s, self.q_heads, self.d_head)
      k = self.wk(x).view(b, s, self.kv_heads, self.d_head)
      v = self.wv(x).view(b, s, self.kv_heads, self.d_head)
      
      q,k = rope_apply(q, rope_freq), rope_apply(k, rope_freq)
      q,k,v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
      
      # query -> (b, q_head, s, d_h)
      #key -> (b, kv_heads, s, d_h)
      q = q.view(b, self.query_groups, self.kv_heads, s, self.d_head)
      attn_scores = q @ k.transpose(2,3).unsqueeze(1) #key -> (b,1,kv_heads,d,s)
      if mask is not None:
        attn_scores = attn_scores.masked_fill(mask ==0, -torch.inf)
      attn_scores = torch.softmax((attn_scores * self.scale), dim=-1)
      
      out = (attn_scores @ v.unsqueeze(1)).transpose(1,2).contiguous().view(b,s,d)
      return self.wo(out)
      
      
# x =  torch.randn(2,3,4)
# attn = GQA_SA(4,2,4,2)
# i = torch.arange(0,3).view(3,1)
# j = torch.arange(0,3).view(1,3)
# mask = (i>=j) & (j >= i-2+1)
# print(attn(x,mask = mask).shape)
# print(attn(x))
      