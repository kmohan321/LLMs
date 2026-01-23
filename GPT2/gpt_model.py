import torch
import torch.nn as nn
from .attention import MHA
from common import LAYER_NORM, FFN

#official approximation for GELU
class GELU(nn.Module):
  def __init__(self):
    super().__init__()
    
  def forward(self,x):
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2/torch.pi)) * (x + 0.044715 * x**3)))
  
class TRANSFOMER_BLOCK(nn.Module): 
  def __init__(self, config):
    super().__init__()
    
    self.dropout = nn.Dropout(config['drop'])
    
    # attention part
    self.attention_norm = LAYER_NORM(config['hidden_dim'], config['eps'])
    self.attention = MHA(config['hidden_dim'], config['heads'], config['drop'], config['bias'])
    
    # ffn part
    self.ffn_norm = LAYER_NORM(config['hidden_dim'], config['eps'])
    self.ffn = FFN(config['hidden_dim'], GELU(), config['mlp_multiplier'])
    
  def forward(self,x, past_kv):
    
    out, k, v = self.attention(self.attention_norm(x), past_kv)
    x = x + out
    x = x + self.ffn(self.ffn_norm(x))
    return x, k, v
  
class GPT(nn.Module):
  """
  Args:
        hidden_dim (int): Size of the hidden layer representations.
        vocab_size (int): The total number of unique tokens in the vocabulary.
        context_length (int): Number of tokens the model processes in one forward pass (sequence length).
        heads (int): Number of attention heads in the multi-head attention mechanism.
        blocks (int): Number of Transformer blocks (stacked layers) in the model.
        drop (float, optional): Dropout rate to prevent overfitting. Default is 0.2.
        bias (bool, optional): Whether to include bias terms in linear layers. Default is False.
    """
  def __init__(self, config):
    super().__init__()
    
    self.token_embeddings = nn.Embedding(config["vocab_size"], config["hidden_dim"])
    self.pos_embeddings = nn.Embedding(config["context_length"],config["hidden_dim"])
    self.dropout = nn.Dropout(config["drop"])
    
    self.blocks = nn.ModuleList([TRANSFOMER_BLOCK(config) for _ in range(config["blocks"])])
    self.model_norm = LAYER_NORM(config['hidden_dim'], config['eps'])
    self.final_layer = nn.Linear(config["hidden_dim"],config["vocab_size"], config["bias"])
    
  def forward(self, x, past_kv):
    
    if past_kv is not None:
      seq_length = past_kv[0][0].shape[2]
      pos_embedds = self.pos_embeddings(torch.tensor(seq_length-1, device=x.device))
    else:
      seq_length = x.shape[1]
      pos_embedds = self.pos_embeddings(torch.arange(seq_length, device=x.device))
      
    token_embedds = self.token_embeddings(x)
    x = token_embedds + pos_embedds
    x = self.dropout(x)
    
    present_kv = []
    for idx, block in enumerate(self.blocks):
      if past_kv is None:
        kv_set = past_kv
      else:
        kv_set = past_kv[idx]
      x, k, v = block(x, kv_set)
      present_kv.append((k,v))
      
    x = self.model_norm(x)
    x = self.final_layer(x)
    return x, present_kv


   
    
    
    
    
