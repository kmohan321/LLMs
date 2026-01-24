import torch
import torch.nn as nn
from .moe_shared import DeepSeekMoE
from .mla import MLA
from common import RMSE_NORM, Frequencies

class Block(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    
    self.attention = MLA(cfg['hidden_dim'], cfg['heads'], cfg['kv_compressed_dim'], cfg['q_compressed_dim'],
                         cfg['qk_rope_dim'], cfg['v_head_dim'])
    self.ffn = DeepSeekMoE(cfg['num_shared_exp'], cfg['num_routed_exp'], cfg['hidden_dim'],
                           cfg['intermediate_dim'], cfg['topk'])
    
    self.atten_norm = RMSE_NORM(cfg['hidden_dim'], cfg['eps'])
    self.ffn_norm = RMSE_NORM(cfg['hidden_dim'], cfg['eps'])
    
  def forward(self, x, freq, mask):
    x = x + self.attention(self.atten_norm(x), freq, mask)
    x = x + self.ffn(self.ffn_norm(x))
    return x

class DeepSeek(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    
    self.embedding = nn.Embedding(cfg['vocab_size'], cfg['hidden_dim'])
    self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg['num_blocks'])])
    self.freq = Frequencies(cfg['max_seq_length'], cfg['qk_rope_dim'])
    
    self.model_norm = RMSE_NORM(cfg['hidden_dim'], cfg['eps'])
    self.final_layer = nn.Linear(cfg['hidden_dim'], cfg['vocab_size'])
    
  def forward(self, x):
    seq_len = x.shape[1]
    mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
    
    x = self.embedding(x)
    for block in self.blocks:
      x = block(x, self.freq.complex_freq, mask)
      
    x = self.model_norm(x)
    return self.final_layer(x)
  
# cfg ={
#   'hidden_dim': 1024,
#   'num_blocks': 3,
#   'vocab_size': 30000,
#   'eps': 1e-5,
#   'max_seq_length': 40000,
#   'num_shared_exp': 2,
#   'num_routed_exp': 2,
#   'intermediate_dim': 512,
#   'topk': 1,
#   'heads': 16,
#   'kv_compressed_dim': 32,
#   'q_compressed_dim': 16,
#   'qk_rope_dim': 8,
#   'v_head_dim': 256
# }
