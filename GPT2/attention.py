import torch
import torch.nn as nn


class SELF_ATTENTION(nn.Module):
  def __init__(self,
               hidden_dim,
               context_length,
               heads,drop = 0.5,
               bias = False):
    super().__init__()
    
    self.heads = heads
    self.hidden_dim = hidden_dim
    
    # checking hidden_dim div by heads
    assert (hidden_dim%self.heads==0), 'hidden_dim is not div by heads'
    
    #preparing the weight matrices
    self.query_weights = nn.Linear(hidden_dim,hidden_dim,bias=bias)
    self.key_weights =  nn.Linear(hidden_dim,hidden_dim,bias=bias)
    self.value_weights = nn.Linear(hidden_dim,hidden_dim,bias=bias)
    
    self.drop = nn.Dropout(drop)

    self.head_dim = hidden_dim//self.heads
    
    # creating register_buffer(for device management)
    self.register_buffer(
      'masking',
      torch.triu(torch.ones(context_length,context_length),diagonal=1)
    )
    self.final_linear = nn.Linear(hidden_dim,hidden_dim)
    
  def forward(self,inputs):
    
    batch, tokens , d_length = inputs.shape
    
    querys = self.query_weights(inputs)
    keys = self.key_weights(inputs)
    values = self.value_weights(inputs)
    
    #splitting weights for heads
    query_heads = querys.view(batch,tokens,self.heads,self.head_dim)
    key_heads = keys.view(batch,tokens,self.heads,self.head_dim)
    value_heads = values.view(batch,tokens,self.heads,self.head_dim)
    
    query_heads = query_heads.transpose(1,2)
    key_heads = key_heads.transpose(1,2)
    value_heads = value_heads.transpose(1,2)
    
    #calculating the attention weights
    
    attention_scores = torch.matmul(query_heads,key_heads.transpose(2,3))
    attention_scores.masked_fill_(self.masking.bool()[:tokens,:tokens], -torch.inf)
    attention_weights = torch.softmax(attention_scores/d_length**0.5,dim=-1)
    
    #prevent overfitting or too much relying 
    attention_weights = self.drop(attention_weights)
    context_vecs = attention_weights @ value_heads
    context_vecs = context_vecs.transpose(1,2)
    context_vecs = context_vecs.contiguous().view(batch,tokens,self.hidden_dim)
    
    return self.final_linear(context_vecs)


    
    
    
    
    
    