import torch
import torch.nn as nn
from Modules import Embeddings,Encoder

class BERT(nn.Module):
  def __init__(self,
               hidden_dim:int,
               vocab_size:int,
               max_seq_length:int,
               heads:int,
               drop_mha:float,
               drop_ffn:float,
               ffn_multiplier:int,
               drop:float,
               encoder_layers:int
               ):
    super().__init__()
    
    self.embeddings = Embeddings(hidden_dim,vocab_size,max_seq_length)
    self.encoder_layers = nn.ModuleList([Encoder(heads,hidden_dim,drop_mha,drop_ffn,ffn_multiplier,drop) 
                                         for _ in range(encoder_layers)])
    
  def forward(self,x,segment_info):
    # creating mask
    # (b,s) -> (b,1,s,s)
    mask = (x>0).unsqueeze(1).repeat(1,x.shape[1],1).unsqueeze(1)
    
    x = self.embeddings(x,segment_info)
    
    for layer in self.encoder_layers:
      x = layer(x,mask)
    
    return x
  

    