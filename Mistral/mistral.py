import torch
import torch.nn as nn
from .gqa_sa import GQA_SA
from common import Frequencies, RMSE_NORM, FFN_SW
  
class Attention_Block(nn.Module):
  def __init__(self,config):
    super().__init__()
    
    self.atten = GQA_SA(config["num_heads_q"],  config["num_heads_kv"], config["hidden_dims"])
    self.atten_norm = RMSE_NORM(config["hidden_dims"], config["eps"])
    self.ffn_norm = RMSE_NORM(config["hidden_dims"], config["eps"])
    self.ffn = FFN_SW(config["hidden_dims"], config["intermediate_dim"])
    
  def forward(self,x, freq, mask):
    x = x + self.atten(self.atten_norm(x), freq, mask)
    x = x + self.ffn(self.ffn_norm(x))
    return x
  
class Mistral(nn.Module):
  def __init__(self,config):
    super().__init__()    
    
    self.blocks = nn.ModuleList([Attention_Block(config) for _ in range(config["num_blocks"])])
    self.embedding = nn.Embedding(config["vocab_size"],config["hidden_dims"])
    self.final_layer = nn.Linear(config["hidden_dims"],config["vocab_size"],bias=False)
    self.model_norm = RMSE_NORM(config["hidden_dims"],config["eps"])
    
    self.freq = Frequencies(config["seq_length"], config["head_dim"])
    self.w = config["window_size"]
    
  def forward(self, x):
    s = x.shape[1]
    
    x = self.embedding(x)
    
    mask = torch.full((s,s), -torch.inf)
    casual_mask = torch.triu(torch.ones(s,s), diagonal=1)
    window_mask = torch.tril(torch.ones(s,s), diagonal=-self.w)
    final_mask = (casual_mask==1) | (window_mask ==1)
    mask = mask.masked_fill(final_mask, -torch.inf).masked_fill(~final_mask, 0.0)
    print(mask)
    
    for block in self.blocks:
      x = block(x, self.freq.complex_freq, mask)
    
    x = self.model_norm(x)
    return self.final_layer(x)
      
# config = {
#     "num_blocks": 2,
#     "hidden_dims": 256,
#     "num_heads_q": 8,
#     "num_heads_kv": 4,
#     "seq_length": 4096,
#     "intermediate_dim": 512,
#     "vocab_size": 52000,
#     "eps": 1e-5,
#     "head_dim" : 32,
#     "window_size" : 256
    
# }