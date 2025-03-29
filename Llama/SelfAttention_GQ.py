import torch
import torch.nn as nn
from Rotatory import Frequencies

class SELF_ATTENTION(nn.Module):
  def __init__(self,
               hidden_dims :int,
               num_heads_q:int,
               num_heads_kv :int,
               seq_length :int):
    super().__init__()
    
    self.num_heads_q = num_heads_q
    self.num_heads_kv = num_heads_kv
    
    assert hidden_dims % self.num_heads_q==0, 'hidden_dim must be divisible by num_heads'
    assert self.num_heads_q % self.num_heads_kv == 0, "num_heads_q must be divisible by num_heads_kv"
    
    self.head_dim = hidden_dims//self.num_heads_q
    self.groups = self.num_heads_q//self.num_heads_kv
    
    #head_dim remains constant
    self.wq = nn.Linear(hidden_dims,num_heads_q*self.head_dim,bias=False)
    self.wk = nn.Linear(hidden_dims,num_heads_kv*self.head_dim,bias=False) 
    self.wv = nn.Linear(hidden_dims,num_heads_kv*self.head_dim,bias=False)
    
    self.rotatory = Frequencies(seq_length,self.head_dim)
    #buffer for mask
    self.register_buffer(
      "mask",
      torch.tril(torch.ones(seq_length,seq_length,dtype=torch.bool))
    )
    #output layer
    self.out_layer = nn.Linear(self.num_heads_q*self.head_dim,hidden_dims,bias=False)
  
  def forward(self, x:torch.Tensor,is_causal = True):
    
    b, s, d = x.shape
    query = self.wq(x).view(b,s,self.num_heads_q,self.head_dim)
    key = self.wk(x).view(b,s,self.num_heads_kv,self.head_dim)
    value = self.wv(x).view(b,s,self.num_heads_kv,self.head_dim).transpose(1,2)
    
    #(b,s,h,d) -> (b,h,s,d) 
    rotated_query = self.rotatory(query).transpose(1,2) #applying the rope here
    rotated_key = self.rotatory(key).transpose(1,2)

    # Group query heads to match key heads
    #(B,Hq,S,D) -> (B,Hkv,G,S,D)
    rotated_query = rotated_query.view(b,self.num_heads_kv,self.groups,s,self.head_dim)

    attention_score = rotated_query @ rotated_key.transpose(2,3).unsqueeze(2)
    if is_causal:
      attention_score = torch.masked_fill(attention_score, mask=self.mask[:s,:s]==0,value = -torch.inf)
    attention_weights = torch.softmax(attention_score/self.head_dim**0.5,dim=-1)
    
    out = attention_weights @ value.unsqueeze(2)
    #(b,hkv,G,S,D) -> (b,hq,S,D) ->(b,S,hq,D)
    out = out.view(b,self.num_heads_kv*self.groups,s,self.head_dim).transpose(1,2)
    out = out.contiguous().view(b,s,d)
    
    return self.out_layer(out)
  