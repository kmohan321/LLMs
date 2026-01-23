import torch
import torch.nn as nn
from common import RMSE_NORM, Frequencies
from .atten_gqa_kv import GQA
from .moe import SparseMOE

class TransformerBlock(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    
    self.norm1 = RMSE_NORM(cfg['hidden_dim'], cfg['eps'])
    self.norm2 = RMSE_NORM(cfg['hidden_dim'], cfg['eps'])
    
    self.gqa_atten = GQA(
      cfg['num_kv_heads'],
      cfg['num_q_heads'],
      cfg['hidden_dim'],
      cfg['head_dim'],
      cfg['qk_norm'],
      cfg['causal']
    )
    
    self.ffn = SparseMOE(
      cfg['num_experts'],
      cfg['hidden_dim'],
      cfg['intermediate_dim'],
      cfg['num_exp_tokens']
    )
    
  def forward(self, x, freq, past_kv=None):
    y, k_new, v_new = self.gqa_atten(self.norm1(x), freq, past_kv)
    x = x + y
    x = x + self.ffn(self.norm2(x))
    return x, k_new, v_new
  
class QWEN3_MOE(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    
    self.embedding = nn.Embedding(cfg['vocab_size'], cfg['hidden_dim'])
    
    self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg['num_blocks'])])
    
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
