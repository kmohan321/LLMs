import torch
import torch.nn as nn
from .attention_gqa import GQA
from common import RMSE_NORM, FFN, Frequencies
  
class Attention_Block(nn.Module):
  def __init__(self,config):
    super().__init__()
    
    self.attention = GQA(config["hidden_dims"], config["num_heads_q"], config["num_heads_kv"])
    self.attention_norm = RMSE_NORM(config["hidden_dims"], config["eps"])
    self.ffn_norm = RMSE_NORM(config["hidden_dims"], config["eps"])
    self.ffn = FFN(config["hidden_dims"], config["intermediate_size"])
    
  def forward(self,x, freq, mask):
    x = x + self.attention(self.attention_norm(x), freq, mask)
    x = x + self.ffn(self.ffn_norm(x))
    return x
  
class LLama_Basic(nn.Module):
  def __init__(self,config):
    super().__init__()    
    
    head_dim = config['hidden_dim']//config['num_heads_q']
    
    self.blocks = nn.ModuleList([Attention_Block(config) for _ in range(config["num_blocks"])])
    self.embedding = nn.Embedding(config["vocab_size"],config["hidden_dims"])
    self.final_layer = nn.Linear(config["hidden_dims"],config["vocab_size"],bias=False)
    self.model_norm = RMSE_NORM(config["hidden_dims"],config["eps"])
    self.freq = Frequencies(config['seq_length'], head_dim)
    
  def forward(self,tokens):
    s = tokens.shape[1]
    mask = torch.tril(torch.ones(s,s, dtype=torch.bool, device= tokens.device))
    x = self.embedding(tokens)
    
    for block in self.blocks:
      x = block(x, self.freq.complex_freq, mask)
    
    x = self.model_norm(x)
    return self.final_layer(x)
      
  