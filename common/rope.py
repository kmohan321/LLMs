import torch
import torch.nn as nn

"Class for computing the frequencies"
class Frequencies(nn.Module):
  def __init__(self, seq_length,head_dim):
    super().__init__()
    
    assert head_dim%2==0, 'head_dim should be even'
    half_dim = head_dim // 2
    
    # m values for different positions in sequence
    m = torch.arange(0,seq_length)
    
    # theta values for different index in token embedding
    theta = 10000 ** (- torch.arange(half_dim) / half_dim)
    freq = torch.outer(m,theta) #shape->(m,d/2)
    complex_freq = torch.polar(torch.ones_like(freq),freq)
    self.register_buffer('complex_freq', complex_freq.unsqueeze(0).unsqueeze(1))

"Function to apply rope"
def apply_rope(x:torch.Tensor, complex_freq: torch.Tensor):
  
  b ,h, s, d = x.shape
  x1 = x[...,:d//2]
  x2 = x[..., d//2:]
  x = torch.stack([x1,x2], dim = -1)
  x = torch.view_as_complex(x)
  x = x * complex_freq[:, :, :s]
  x = torch.view_as_real(x)
  x1 = x[..., 0]
  x2 = x[..., 1]
  x = torch.cat([x1, x2], dim =-1)
  x = x.view(b,h,s,d)
  return x