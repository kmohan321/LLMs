import torch
import torch.nn as nn

class MHA(nn.Module):
  def __init__(self,
               hidden_dim,
               heads,
               drop = 0.5,
               bias = False):
    super().__init__()
    
    self.heads = heads
    self.hidden_dim = hidden_dim
    
    assert (hidden_dim % self.heads==0), 'hidden_dim is not div by heads'
    self.head_dim = hidden_dim//self.heads
    
    self.wqkv = nn.Linear(hidden_dim, 3 * self.head_dim * self.heads ,bias=bias)
    
    self.drop = nn.Dropout(drop)
    self.wo = nn.Linear(self.head_dim * self.heads, hidden_dim)
    self.scale = self.head_dim ** -0.5
    
  def forward(self, x, past_kv = None):
    
    b, s, d = x.shape
    
    q, k, v = torch.chunk(self.wqkv(x), 3, dim = -1)
    
    q = q.view(b, s, self.heads, self.head_dim).transpose(1,2)
    k = k.view(b, s, self.heads, self.head_dim).transpose(1,2)
    v = v.view(b, s, self.heads, self.head_dim).transpose(1,2)
    
    if past_kv is not None:
      k = torch.cat([past_kv[0], k], dim=2)
      v = torch.cat([past_kv[1], v], dim=2)
      
    seq_len = k.shape[2]
    
    atten_score = torch.matmul(q,k.transpose(2,3))
    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))
    mask = mask[-q.shape[2]:, :]
    atten_score = torch.masked_fill(atten_score, mask=mask==0, value=-torch.inf)
    atten_weights = torch.softmax(atten_score * self.scale, dim=-1)
    
    #prevent overfitting or too much relying 
    atten_weights = self.drop(atten_weights)
    out = atten_weights @ v
    out = out.transpose(1,2)
    out = out.contiguous().view(b, s, self.heads * self.head_dim)
    
    return self.wo(out), k, v


    
    
    
    
    
    