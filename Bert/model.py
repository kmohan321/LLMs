import torch
import torch.nn as nn
from common import FFN, LAYER_NORM
from .modules import Embeddings, MultiHeadAttention

class Block(nn.Module):
    def __init__(self,
        heads:int,
        hidden_dim:int,
        drop_mha:float,
        drop_hidden:float,
        ffn_multiplier:int,
        eps: float
        ):
      super().__init__()
      
      self.attention = MultiHeadAttention(heads,hidden_dim,drop_mha)
      self.ffn = FFN(hidden_dim, nn.GELU(), ffn_multiplier, drop_hidden)
      self.dropout = nn.Dropout(drop_hidden)
      
      self.norm_mha = LAYER_NORM(hidden_dim, eps=eps)
      self.norm_ffn = LAYER_NORM(hidden_dim, eps=eps)
      
    def forward(self,x,mask):
      
      x1 = self.dropout(self.attention(x,mask))
      x = x1 + x                       
      x = self.norm_mha(x)
      x2 = self.dropout(self.ffn(x))
      x  = x2 + x
      return self.norm_ffn(x)

class BERT(nn.Module):
  def __init__(self, config
               ):
    super().__init__()
    
    self.embeddings = Embeddings(config["hidden_dim"],config["vocab_size"],config["max_seq_length"],
                                 config["type_vocab_size"], config["drop_hidden"], config["eps"], config["pad_token_id"])
    self.blocks = nn.ModuleList([Block(config["heads"], config["hidden_dim"], config["drop_mha"], config["drop_hidden"], 
                                       config["ffn_multiplier"], config["eps"]) 
                                         for _ in range(config["blocks"])])
    self.pooler_layer = nn.Linear(config["hidden_dim"], config["hidden_dim"])
    self.act = nn.Tanh()
    
  def forward(self, x, segment_info = None):
    # creating mask
    # (b,s) -> (b,1,s,s)
    mask = (x>0).unsqueeze(1).repeat(1, x.shape[1], 1).unsqueeze(1)
    
    x = self.embeddings(x, segment_info)
    
    for layer in self.blocks:
      x = layer(x, mask)
      
    return x, self.act(self.pooler_layer(x[:, 0]))

    