import torch
import torch.nn as nn
from common import apply_rope

class MLA(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 heads: int,
                 kv_compressed_dim: int,
                 q_compressed_dim: int,
                 qk_rope_dim: int,
                 v_head_dim: int
                 ):
        super().__init__()
        
        assert hidden_dim % heads ==0, "Hidden_dim must be divisible by heads" 
        self.h = hidden_dim
        self.n_h = heads
        self.d_h = hidden_dim // heads
        self.d_ckv = kv_compressed_dim
        self.d_cq = q_compressed_dim 
        self.qk_dr = qk_rope_dim
        self.d_hq = qk_rope_dim + self.d_h
        self.d_hv =  v_head_dim
         
        #up projection and down projection matrices
        self.w_dkv = nn.Linear(self.h, self.d_ckv)
        self.w_ukv = nn.Linear(self.d_ckv, self.n_h * (self.d_h + self.d_hv)) #generally (d_h = d_hv)
        
        self.w_kr = nn.Linear(self.h, 1 * self.qk_dr) #decoupled key for RoPE
        
        self.w_dq = nn.Linear(self.h, self.d_cq)
        self.w_uq = nn.Linear(self.d_cq, self.n_h * self.d_h)
        
        self.w_qr = nn.Linear(self.d_cq, self.n_h * self.qk_dr) 

        self.wo = nn.Linear(self.n_h *  self.d_hv, hidden_dim)
        self.scale = self.d_hq ** -0.5
        
    def forward(self, x : torch.Tensor, freq : torch.Tensor, mask : torch.Tensor | None = None):

        b,s,d = x.shape
        #(b,s,d) -> (b,s,d_cq)
        c_q = self.w_dq(x)
        
        #(b,s,d_cq) -> (b,s,n_h * d_h)
        q_c = self.w_uq(c_q).view(b,s,self.n_h,self.d_h)
        q_r = self.w_qr(c_q).view(b,s,self.n_h,self.qk_dr)
        
        q_r = q_r.transpose(1,2)
        q_c = q_c.transpose(1,2)
        q_r = apply_rope(q_r,freq) 
        
        #(b,n_h,s,d_h),(b,n_h,s,d_r) ->(b,n_h,s,d_hq)
        q = torch.cat([q_c,q_r],dim=-1)
        
        #(b,s,d) -> (b,s,d_ckv)
        c_kv = self.w_dkv(x) #can be cached
        
        kv_c = self.w_ukv(c_kv).view(b,s,self.n_h,-1)
        k_c, v_c = torch.split(kv_c, [self.d_h, self.d_hv], dim=-1)
        k_c = k_c.transpose(1,2)
        v_c = v_c.transpose(1,2)
        
        k_r = self.w_kr(x).view(b,s,1,self.qk_dr)   # can be cached 
        k_r = k_r.transpose(1,2)
        k_r = apply_rope(k_r,freq)
        k_r = k_r.expand(-1, self.n_h, -1, -1)
        k = torch.cat([k_c,k_r],dim=-1)
        
        attn_score = (q @ k.transpose(2,3)) * self.scale
        if mask is not None:
            attn_score = attn_score.masked_fill(mask==0,-torch.inf)
        attn_weights = torch.softmax(attn_score, dim=-1)
        out = (attn_weights @ v_c).transpose(1,2).contiguous().view(b,s,self.n_h*self.d_hv) 
        return self.wo(out)
