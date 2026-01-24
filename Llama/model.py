import torch
import torch.nn as nn
from .attention_gqa import GQA
from common import RMSE_NORM, FFN_SW, Frequencies
  
class Attention_Block(nn.Module):
  def __init__(self,config):
    super().__init__()
    
    self.attention = GQA(config["hidden_dims"], config["num_heads_q"], config["num_heads_kv"])
    self.attention_norm = RMSE_NORM(config["hidden_dims"], config["eps"])
    self.ffn_norm = RMSE_NORM(config["hidden_dims"], config["eps"])
    self.ffn = FFN_SW(config["hidden_dims"], config["intermediate_size"])
    
  def forward(self,x, freq, past_kv):
    y, k, v = self.attention(self.attention_norm(x), freq, past_kv)
    x = x + y
    x = x + self.ffn(self.ffn_norm(x))
    return x, k, v
  
class LLama_Basic(nn.Module):
  def __init__(self,config):
    super().__init__()    
    
    head_dim = config['hidden_dims']//config['num_heads_q']
    
    self.blocks = nn.ModuleList([Attention_Block(config) for _ in range(config["num_blocks"])])
    self.embedding = nn.Embedding(config["vocab_size"],config["hidden_dims"])
    self.final_layer = nn.Linear(config["hidden_dims"],config["vocab_size"],bias=False)
    self.model_norm = RMSE_NORM(config["hidden_dims"],config["eps"])
    self.freq = Frequencies(config['seq_length'], head_dim)
    
  def forward(self, tokens, past_kv = None):
    
    x = self.embedding(tokens)
    
    present_kv = []
    for idx, block in enumerate(self.blocks):
      if past_kv is not None:
        kv_set = past_kv[idx]
      else:
        kv_set = past_kv
      x, k, v = block(x, self.freq.complex_freq, kv_set)
      present_kv.append((k,v))
    
    x = self.model_norm(x)
    return self.final_layer(x), present_kv
      
  