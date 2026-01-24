import torch
import torch.nn as nn

class Expert(nn.Module):
  def __init__(self,
                hidden_dim: int,
                intermediate_dim: int
                 ):
    super().__init__()
    
    self.w1 = nn.Linear(hidden_dim, intermediate_dim)
    self.gated = nn.Linear(hidden_dim, intermediate_dim)
    self.act = nn.GELU()
    self.w2 = nn.Linear(intermediate_dim, hidden_dim)
      
  def forward(self, x : torch.Tensor):
    y = self.act(self.gated(x))
    x = y * self.w1(x)
    x = self.w2(x)
    return x
    
# two types of experts one is shared and the other routed expert
class DeepSeekMoE(nn.Module):
  def __init__(self,
               num_shared_exp: int,
               num_routed_exp: int,
               hidden_dim: int,
               intermediate_dim: int,
               topk: int
               
               ):
     super().__init__()
     self.k = topk
     self.s_experts = num_shared_exp
     
     self.w_affinity = nn.Linear(hidden_dim,num_routed_exp,bias=False) #for calculating the affinity score
     
     self.shared_experts = nn.ModuleList([Expert(hidden_dim, intermediate_dim) for _ in range(num_shared_exp)])
     self.routed_experts = nn.ModuleList([Expert(hidden_dim, intermediate_dim) for _ in range(num_routed_exp)])
     
  def forward(self, x: torch.Tensor):
    b,s,d = x.shape
    
    affinity_score = torch.sigmoid(self.w_affinity(x))
    topk_probs, topk_indicies = torch.topk(affinity_score, k=self.k, dim=-1)
    
    #normalising the topk_probs 
    topk_probs = topk_probs / torch.sum(topk_probs, dim=-1, keepdim=True)
    
    #for routed experts
    r_expert_out = torch.zeros(b, s, self.k, d, device=x.device)
    for idx, r_expert in enumerate(self.routed_experts):
        mask = topk_indicies == idx
        if mask.any():
          indicies = torch.where(mask)
          
          tokens = x[indicies[0],indicies[1]] #tokens for this expert (b*t,d)
          out = topk_probs[indicies[0],indicies[1],indicies[2]].unsqueeze(-1) * r_expert(tokens)
          r_expert_out[indicies[0],indicies[1],indicies[2]] = out
     
    s_expert_out = torch.zeros(b,s,self.s_experts,d, device=x.device)
    for idx, s_expert in enumerate(self.shared_experts):
      s_expert_out[:,:,idx] = s_expert(x)
      
    final_out = x + torch.sum(s_expert_out,dim=2) + torch.sum(r_expert_out,dim=2)
    return final_out
