import torch
import torch.nn as nn
from .atten_gqa_kv import GQA
from common import FFN_SW, RMSE_NORM, Frequencies

class Block(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    
    self.attention_norm = RMSE_NORM(cfg['hidden_dim'], cfg['eps'])
    self.ffn_norm = RMSE_NORM(cfg['hidden_dim'], cfg['eps'])
    
    self.attention = GQA(
      cfg['num_kv_heads'],
      cfg['num_q_heads'],
      cfg['hidden_dim'],
      cfg['head_dim'],
      cfg['qk_norm'],
      cfg['causal']
    )
    
    self.ffn = FFN_SW(
      cfg['hidden_dim'],
      cfg['intermediate_dim']
    )
    
  def forward(self, x, freq, past_kv=None):
    y, k_new, v_new = self.attention(self.attention_norm(x), freq, past_kv)
    x = x + y
    x = x + self.ffn(self.ffn_norm(x))
    return x, k_new, v_new
      
class QWEN3(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    
    self.embedding = nn.Embedding(cfg['vocab_size'], cfg['hidden_dim'])
    self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg['num_blocks'])])
    self.model_norm  = RMSE_NORM(cfg['hidden_dim'], cfg['eps'])
    self.final_layer = nn.Linear(cfg['hidden_dim'], cfg['vocab_size'], bias=False)
    
    self.freq = Frequencies(cfg['max_seq_len'], cfg['head_dim'])
    
  def forward(self, token_ids, past_kv=None):
    x = self.embedding(token_ids)
    
    present_kv = []
    for i, block in enumerate(self.blocks):
      
      if past_kv is not None:
        kv_set = past_kv[i]
      else:
        kv_set = past_kv
      x, k_new, v_new = block(x, self.freq.complex_freq, kv_set)
      
      present_kv.append((k_new, v_new))
      
    x = self.model_norm(x)
    x = self.final_layer(x)
    return x, present_kv