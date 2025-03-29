import torch
import torch.nn as nn

class Expert(nn.Module):
  def __init__(self,
                hidden_dim: int,
                mlp_multiplier:int
                 ):
      super().__init__()
      
      self.expert = nn.Sequential(
        nn.Linear(hidden_dim, mlp_multiplier * hidden_dim),
        nn.GELU(),
        nn.Linear(mlp_multiplier * hidden_dim, hidden_dim)
    )
  def forward(self, x : torch.Tensor):
      return self.expert(x)
    
class DenseMOE(nn.Module):
  def __init__(self,
               num_experts :int,
               hidden_dim : int,
               mlp_multiplier : int
               ):
    super().__init__()
    self.router = nn.Linear(hidden_dim,num_experts,bias=False) #multiplying by weight only
    self.expert_layer = nn.ModuleList([Expert(hidden_dim,mlp_multiplier) for _ in range(num_experts)])
    
  def forward(self, x: torch.Tensor):
    
    #(b,s,d) -> (b,s,num_experts)
    expert_weights = torch.softmax(self.router(x),dim=-1)
    
    #(b,s,d) -> (b,s,num_experts,d)
    stacked_out = torch.stack([expert(x) for expert in self.expert_layer],dim=2)

    out = stacked_out * expert_weights.unsqueeze(-1) #(b,s,num_experts,d)
    out = torch.sum(out,dim=2) #(b,s,d)
    return out
  
class SparseMOE(nn.Module):
  def __init__(self,
              num_experts :int,
              hidden_dim : int,
              mlp_multiplier : int,
              k : int
               ):
    super().__init__()
    assert k <= num_experts, "k is larger than num_experts"
    self.k = k
    self.num_experts = num_experts

    self.router = nn.Linear(hidden_dim,num_experts,bias=False) #multiplying by weight only
    self.expert_layer = nn.ModuleList([Expert(hidden_dim,mlp_multiplier) for _ in range(num_experts)])
    
  def forward(self,x : torch.Tensor):
    
    # (b,s,d) -> (b,s,k)
    token_probs, probs_idx = torch.topk(torch.softmax(self.router(x),dim=-1),k = self.k, dim=-1)
    
    output = torch.zeros_like(x).unsqueeze(2).expand(-1, -1, self.k, -1) #(b,s,k,d)
    
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
      
