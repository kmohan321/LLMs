import torch
import torch.nn as nn
from common import LAYER_NORM

class MultiHeadAttention(nn.Module):
  def __init__(self,
        heads:int,
        hidden_dim:int,
        drop_mha:float
        ):
    super().__init__()
    
    self.heads = heads
    self.hidden_dim = hidden_dim
    assert self.hidden_dim % self.heads == 0,'hidden_dim must be divisible by heads'
    
    self.head_dim = self.hidden_dim//self.heads
    
    self.wq = nn.Linear(self.hidden_dim,self.hidden_dim)
    self.wk = nn.Linear(self.hidden_dim,self.hidden_dim)  
    self.wv = nn.Linear(self.hidden_dim,self.hidden_dim)  
    self.wo = nn.Linear(self.hidden_dim,self.hidden_dim)
    
    self.scale = self.head_dim ** -0.5
    self.dropout = nn.Dropout(p=drop_mha)
    
  def forward(self, x, mask):
    b,s,d = x.shape
    # (b,s,d) -> (b,s,h,h_d)
    query = self.wq(x).view(-1,s,self.heads,self.head_dim)
    key = self.wk(x).view(-1,s,self.heads,self.head_dim)
    value = self.wv(x).view(-1,s,self.heads,self.head_dim)
    
    # (b,s,h,h_d) -> (b,h,s,h_d)
    query,key,value = query.transpose(1,2),key.transpose(1,2),value.transpose(1,2)
    
    # (b,h,s,h_d) -> (b,h,s,s)
    attention_scores = ((query @ key.transpose(2,3)) * self.scale)
    
    mask = (1.0 - mask.int()) * -10000.0
    attention_scores += mask
    
    # attention_scores = attention_scores.masked_fill(mask==0,-torch.inf) #masking 
    attention_weights = torch.softmax(attention_scores,dim=-1)
    out = self.dropout(attention_weights) @ value
    
    # (b,h,s,h_d) -> (b,s,h,h_d)
    out = out.transpose(1,2)
    out = out.contiguous().view(-1,s,self.hidden_dim)
    return self.wo(out)

# special embedding class for the bert 
# class SinusoidalEmbedding(nn.Module):
#     def __init__(self,
#         hidden_dim:int,
#         max_seq_length:int
#         ):
#       super().__init__()
      
#       self.frequency = torch.exp(-torch.log(torch.tensor(10000))* torch.arange(0,hidden_dim,2) / hidden_dim )
#       seq_idx = torch.arange(max_seq_length).unsqueeze(1)
#       sin_values = torch.sin(seq_idx*self.frequency)
#       cos_values = torch.cos(seq_idx*self.frequency)
#       self.embeddings = torch.cat([sin_values,cos_values],dim=-1)
      
#     def forward(self):
#       return self.embeddings
     
class Embeddings(nn.Module):
    def __init__(self,
        hidden_dim:int,
        vocab_size:int,
        max_seq_length:int,
        type_vocab_size: int,
        drop_hidden: float,
        eps: float,
        pad_token_id: int
        ):
      super().__init__()
      
      #padding_idx doesn't contribute to gradients
      # for tokens (b,s) -> (b,s,d)
      self.word_embeddings = nn.Embedding(vocab_size, hidden_dim,padding_idx=pad_token_id)
      # for sentence segmentation (b,s) -> (b,s,d)
      self.token_type_embeddings = nn.Embedding(type_vocab_size,hidden_dim,padding_idx=pad_token_id)
      # self.position_embeddings = SinusoidalEmbedding(hidden_dim, max_seq_length)
      self.position_embeddings = nn.Embedding(max_seq_length, hidden_dim)
      self.dropout = nn.Dropout(drop_hidden)
      self.layer_norm = LAYER_NORM(hidden_dim, eps)
      
      
    def forward(self, x, segment_info = None):
      
      if segment_info is None:
            segment_info = torch.zeros_like(x)
            
      seq_len = x.shape[1]
      input_ids = torch.arange(0, seq_len, device=x.device)
      
      # (b,s) -> (b,s,d)
      x = self.word_embeddings(x) + self.token_type_embeddings(segment_info) +  self.position_embeddings(input_ids)
      return self.dropout(self.layer_norm(x))