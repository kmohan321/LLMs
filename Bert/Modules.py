import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
  def __init__(self,
        heads:int,
        hidden_dim:int,
        drop_mha:float
        ):
    super().__init__()
    
    self.heads = heads
    self.hidden_dim = hidden_dim
    assert self.hidden_dim%self.heads == 0,'hidden_dim must be divisible by heads'
    
    self.head_dim = self.hidden_dim//self.heads
    
    self.Wq = nn.Linear(self.hidden_dim,self.hidden_dim)
    self.Wk = nn.Linear(self.hidden_dim,self.hidden_dim)  
    self.Wv = nn.Linear(self.hidden_dim,self.hidden_dim)  
    self.Wo = nn.Linear(self.hidden_dim,self.hidden_dim)  #output layer
    
    self.scale = self.head_dim ** -0.5
    self.dropout = nn.Dropout(p=drop_mha)
    
  def forward(self,x,mask):
    b,s,d = x.shape
    # (b,s,d) -> (b,s,h,h_d)
    query = self.Wq(x).view(-1,s,self.heads,self.head_dim)
    key = self.Wk(x).view(-1,s,self.heads,self.head_dim)
    value = self.Wv(x).view(-1,s,self.heads,self.head_dim)
    
    # (b,s,h,h_d) -> (b,h,s,h_d)
    query,key,value = query.transpose(1,2),key.transpose(1,2),value.transpose(1,2)
    
    # (b,h,s,h_d) -> (b,h,s,s)
    attention_scores = ((query @ key.transpose(2,3)) * self.scale)
    
    attention_scores = attention_scores.masked_fill(mask==0,-torch.inf) #masking 
    attention_weights = torch.softmax(attention_scores,dim=-1)
    out = self.dropout(attention_weights) @ value
    
    # (b,h,s,h_d) -> (b,s,h,h_d)
    out = out.transpose(1,2)
    out = out.contiguous().view(-1,s,self.hidden_dim)
    return self.Wo(out)
  
class FFN(nn.Module):
    def __init__(self,
        hidden_dim:int,
        ffn_multiplier:int,
        drop_ffn:float
        ):
      super().__init__()
      
      self.l1 = nn.Linear(hidden_dim,ffn_multiplier*hidden_dim)
      self.l2 = nn.Linear(hidden_dim * ffn_multiplier,hidden_dim)
      self.dropout = nn.Dropout(p=drop_ffn)
      self.act = nn.GELU()
      
    def forward(self,x):
      x = self.dropout(self.l1(x))
      x = self.act(x)
      return self.l2(x)
    
class Encoder(nn.Module):
    def __init__(self,
        heads:int,
        hidden_dim:int,
        drop_mha:float,
        drop_ffn:float,
        ffn_multiplier:int,
        drop:int
        ):
      super().__init__()
      
      self.mha = MultiHeadAttention(heads,hidden_dim,drop_mha)
      self.ffn = FFN(hidden_dim,ffn_multiplier,drop_ffn)
      self.dropout = nn.Dropout(drop)
      
      self.norm_mha = nn.LayerNorm(hidden_dim,eps=1e-6)
      self.norm_ffn = nn.LayerNorm(hidden_dim,eps=1e-6)
      
    def forward(self,x,mask):
      
      x1 = self.dropout(self.mha(x,mask))
      x = x1 + x                       
      x = self.norm_mha(x)
      x2 = self.dropout(self.ffn(x))
      x  = x2 + x
      return self.norm_ffn(x)
    
class SinusoidalEmbedding(nn.Module):
    def __init__(self,
        hidden_dim:int,
        max_seq_length:int):
      super().__init__()
      
      self.frequency = torch.exp(-torch.log(torch.tensor(10000))* torch.arange(0,hidden_dim,2) / hidden_dim )
      seq_idx = torch.arange(max_seq_length).unsqueeze(1)
      sin_values = torch.sin(seq_idx*self.frequency)
      cos_values = torch.cos(seq_idx*self.frequency)
      self.embeddings = torch.cat([sin_values,cos_values],dim=-1)
      
    def forward(self):
      return self.embeddings.unsqueeze(0)
      
      
class Embeddings(nn.Module):
    def __init__(self,
        hidden_dim:int,
        vocab_size:int,
        max_seq_length:int):
      super().__init__()
      
      #padding_idx doesn't contribute to gradients
      # for tokens (b,s) -> (b,s,d)
      self.token_embeddings = nn.Embedding(vocab_size,hidden_dim,padding_idx=0)
      # for sentence segmentation (b,s) -> (b,s,d)
      self.segment_embeddings = nn.Embedding(3,hidden_dim,padding_idx=0)
      self.positional_embeddings = SinusoidalEmbedding(hidden_dim,max_seq_length)
      self.dropout = nn.Dropout(0.1)
      
    def forward(self,x,segment_info):
      
      # (b,s) -> (b,s,d)
      x = self.token_embeddings(x) + self.segment_embeddings(segment_info) +  self.positional_embeddings()
      return self.dropout(x)
    

      
      
      
      
      
      
    
    

    
    
    
    
    
    
    
    
    
    