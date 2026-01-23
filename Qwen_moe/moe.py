import torch
import torch.nn as nn

class Expert(nn.Module):
  def __init__(self,
                hidden_dim: int,
                intermediate_dim:int
                 ):
      super().__init__()
      
      self.w1 = nn.Linear(hidden_dim, intermediate_dim)
      self.act = nn.GELU()
      self.w2  = nn.Linear(intermediate_dim, hidden_dim)
      self.gated = nn.Linear(hidden_dim, intermediate_dim)
    
  def forward(self, x : torch.Tensor):
      y = self.act(self.gated(x))
      x = y * self.w1(x)
      x = self.w2(x)
      return x
      
class SparseMOE(nn.Module):
  def __init__(self,
              num_experts :int,
              hidden_dim : int,
              intermediate_dim : int,
              k : int
               ):
    super().__init__()
    assert k <= num_experts, "k is larger than num_experts"
    self.k = k
    self.num_experts = num_experts

    self.router = nn.Linear(hidden_dim, num_experts,bias=False) #multiplying by weight only
    self.expert_layer = nn.ModuleList([Expert(hidden_dim, intermediate_dim) for _ in range(num_experts)])
    
  def forward(self, x : torch.Tensor):
    
    # (b,s,d) -> (b,s,k)
    b,s,d = x.shape
    token_probs, probs_idx = torch.topk(torch.softmax(self.router(x),dim=-1),k = self.k, dim=-1)
    
    output = torch.zeros(size=(b, s, self.k, d), device=x.device) #(b,s,k,d)
    
    for expert_idx in range(self.num_experts):
      
      expert = self.expert_layer[expert_idx]
      mask = probs_idx == expert_idx
      
      if mask.any():
        token_indices = torch.where(mask) #three index tensors
        expert_tokens = x[token_indices[0], token_indices[1]] #(batch,expert_tokens,d)
        token_weights = token_probs[token_indices[0], token_indices[1], token_indices[2]].unsqueeze(-1)  # (batch,expert_tokens, 1)
        processed_tokens = expert(expert_tokens) * token_weights
        output[token_indices[0], token_indices[1], token_indices[2]] = processed_tokens
        
    return torch.sum(output,dim=2)
      
