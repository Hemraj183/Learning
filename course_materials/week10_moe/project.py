import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# Sparse Mixture of Experts
# ==========================================
class Expert(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed)
        )
    def forward(self, x): return self.net(x)

class MoE(nn.Module):
    def __init__(self, n_embed, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Create Experts
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        
        # Gating Network (The Router)
        self.gating = nn.Linear(n_embed, num_experts)

    def forward(self, x):
        # x: [batch, seq_len, n_embed]
        
        # 1. Gating Logits
        logits = self.gating(x) # [batch, seq, num_experts]
        
        # 2. Select Top-K Experts
        # scores: [batch, seq, k], indices: [batch, seq, k]
        weights, indices = torch.topk(F.softmax(logits, dim=-1), self.top_k)
        
        # 3. Route tokens to experts
        # For simplicity in this demo, we loop (slow, but clear)
        # In production, we assume sparse matrix multiplication
        
        batch, seq, dim = x.shape
        out = torch.zeros_like(x)
        
        for b in range(batch):
            for s in range(seq):
                for k in range(self.top_k):
                    expert_idx = indices[b, s, k].item()
                    weight = weights[b, s, k]
                    
                    selected_expert = self.experts[expert_idx]
                    expert_out = selected_expert(x[b, s].unsqueeze(0))
                    
                    out[b, s] += weight * expert_out.squeeze(0)
                    
        return out

if __name__ == "__main__":
    moe = MoE(n_embed=32, num_experts=8, top_k=2)
    x = torch.randn(2, 5, 32) # Batch 2, Seq 5, Dim 32
    output = moe(x)
    print(f"MoE Output: {output.shape}")
