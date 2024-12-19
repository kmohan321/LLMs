import torch
import torch.nn as nn
from attention import SELF_ATTENTION

class LAYER_NORM(nn.Module):
  def __init__(self,hidden_dim):
    super().__init__()
    
    self.eps = 1e-5 
    self.shift = nn.Parameter(torch.zeros(hidden_dim))
    self.scale = nn.Parameter(torch.ones(hidden_dim))
    
  def forward(self,x):
    mean = torch.mean(x,-1,keepdim=True)
    var = torch.var(x,-1,keepdim=True,unbiased=False)
    
    x = x-mean/torch.sqrt((var + self.eps))
    return x * self.scale + self.shift
  
class GELU(nn.Module):
  def __init__(self):
    super().__init__()
    
  def forward(self,x):
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2/torch.pi)) * (x + 0.044715 * x**3)))

class FEED_FORWARD(nn.Module):
  def __init__(self,hidden_dim):
    super().__init__()
    
    self.ffn = nn.Sequential(
      nn.Linear(hidden_dim,hidden_dim*4),
      GELU(),
      nn.Linear(4*hidden_dim,hidden_dim)
    )
  def forward(self,x):
    return self.ffn(x)
  
class TRANSFOMER_BLOCK(nn.Module): 
  def __init__(self,hidden_dim,context_length,heads,drop=0.2,bias=False):
    super().__init__()
    
    self.dropout = nn.Dropout(drop)
    
    # attention part
    self.norm1 = LAYER_NORM(hidden_dim=hidden_dim)
    self.mha = SELF_ATTENTION(hidden_dim,context_length,heads,drop,bias)
    
    # ffn part
    self.norm2 = LAYER_NORM(hidden_dim=hidden_dim)
    self.ffn = FEED_FORWARD(hidden_dim)
    
  def forward(self,x):
    
    x1 = self.norm1(x)
    x1 = self.mha(x1)
    x1 = self.dropout(x1)
    x = x + x1
    
    x2 = self.norm2(x)
    x2 = self.ffn(x2)
    x2 = self.dropout(x2)
    x = x + x2
    return x
  
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
  def __init__(self,
               hidden_dim,
               vocab_size,
               context_length,
               heads,blocks,
               drop=0.2,
               bias=False):
    super().__init__()
    
    # converts token_ids to embeddings
    self.token_embeddings = nn.Embedding(vocab_size,hidden_dim)
    # using absolute positional encodings(learnable)
    self.pos_embeddings = nn.Embedding(context_length,hidden_dim)
    self.dropout = nn.Dropout(drop)
    
    self.transfomer_blocks = nn.Sequential(
      *[TRANSFOMER_BLOCK(hidden_dim,context_length,heads,drop=0.2,bias=bias)
       for _ in range(blocks)]
    )
    self.final_norm = LAYER_NORM(hidden_dim)
    self.final_layer = nn.Linear(hidden_dim,vocab_size,bias)
    
  def forward(self,x):
    
    seq_length = x.shape[1]
    pos_embedds = self.pos_embeddings(torch.arange(seq_length,device=x.device))
    token_embedds = self.token_embeddings(x)
    
    x = token_embedds + pos_embedds
    x = self.dropout(x)
    
    x = self.transfomer_blocks(x)
    x = self.final_norm(x)
    x = self.final_layer(x)
    return x


   
    
    
    
    
