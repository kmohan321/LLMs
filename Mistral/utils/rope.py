import torch
import torch.nn as nn

"Class for computing the frequencies"
class Frequencies(nn.Module):
  def __init__(self, seq_length,head_dim):
    super().__init__()
    
    assert head_dim%2==0, 'head_dim should be even'
    
    # m values for different positions in sequence
    m = torch.arange(0,seq_length)
    
    # theta values for different index in token vector
    theta = 1/(10000**(2*torch.arange(0,head_dim//2)/head_dim))
    
    #all possible combinations for m and theta
    freq = torch.outer(m,theta) #shape->(m,d/2)
    
    #converting freq to polar
    complex_freq = torch.polar(torch.ones_like(freq),freq)
    
    self.register_buffer('complex_freq', complex_freq.unsqueeze(0).unsqueeze(2))

"Function to apply rope"
def apply_rope(x:torch.Tensor, complex_freq: torch.Tensor):
  b ,s, h, d = x.shape
  x = x.view(b, s, h, -1, 2)
  x = torch.view_as_complex(x)
  x = x * complex_freq[:,:s]
  x = torch.view_as_real(x)
  x = x.view(b,s,h,d)
  return x