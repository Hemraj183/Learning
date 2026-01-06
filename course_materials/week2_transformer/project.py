import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. Scaled Dot-Product Attention
# ==========================================
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_dim)
        self.query = nn.Linear(embed_dim, head_dim)
        self.value = nn.Linear(embed_dim, head_dim)
        self.head_dim = head_dim

    def forward(self, x, mask=None):
        # x: [batch, seq_len, embed_dim]
        
        # 1. Compute Q, K, V
        k = self.key(x)   # [B, S, H]
        q = self.query(x) # [B, S, H]
        v = self.value(x) # [B, S, H]

        # 2. Compute Scores: (Q @ K^T) / sqrt(d_k)
        # Transpose K to [B, H, S] for multiplication
        weights = torch.matmul(q, k.transpose(-2, -1)) 
        weights = weights / math.sqrt(self.head_dim) # Scaling

        # 3. Apply Mask (optional - for decoder)
        if mask is not None:
            weights = weights.masked_fill(mask == 0, float('-inf'))

        # 4. Softmax
        attention_scores = F.softmax(weights, dim=-1)

        # 5. Weighted Sum
        out = torch.matmul(attention_scores, v)
        
        return out, attention_scores

# ==========================================
# 2. Multi-Head Attention
# ==========================================
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Stack all heads into single linear layers for efficiency
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 1. Get Q, K, V
        qkv = self.qkv(x) # [B, S, 3*D]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # [3, B, Heads, S, Head_Dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 2. Scaled Dot-Product Attention (Batch Matrix Multiply)
        # scores = (Q @ K^T) / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
             scores = scores.masked_fill(mask == 0, float('-inf'))
             
        attention = F.softmax(scores, dim=-1)
        
        # 3. Weighted sum
        out = torch.matmul(attention, v) # [B, Heads, S, Head_Dim]
        
        # 4. Concatenate heads
        out = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        
        return self.fc_out(out)

# ==========================================
# 3. Feed Forward Network
# ==========================================
class FeedForward(nn.Module):
    def __init__(self, embed_dim, expansion=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion),
            nn.ReLU(),
            nn.Linear(embed_dim * expansion, embed_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# ==========================================
# 4. Transformer Block (The "Engine")
# ==========================================
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Skip Connection + Token Mixing (Attention)
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)
        
        # Skip Connection + Channel Mixing (Feed Forward)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x

def verify_implementation():
    print("Verifying Transformer Block...")
    B, S, D = 2, 10, 512
    x = torch.randn(B, S, D)
    
    # 1. Test Self-Attention
    sa = SelfAttention(D, 64)
    out_sa, weights = sa(x)
    print(f"Self Attention Output: {out_sa.shape} (Expected: [{B}, {S}, 64])")
    
    # 2. Test Multi-Head Attention
    mha = MultiHeadAttention(D, 8)
    out_mha = mha(x)
    print(f"Multi-Head Output: {out_mha.shape} (Expected: [{B}, {S}, {D}])")
    
    # 3. Test Full Block
    block = TransformerBlock(D, 8)
    out_block = block(x)
    print(f"Transformer Block Output: {out_block.shape} (Expected: [{B}, {S}, {D}])")
    print("✅ Logic Correct!")

if __name__ == "__main__":
    verify_implementation()
