import torch
import torch.nn as nn
import math

# ==========================================
# LoRA Linear Layer
# ==========================================
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=16):
        super().__init__()
        # 1. Frozen Pre-trained Weights
        self.pretrained = nn.Linear(in_features, out_features)
        self.pretrained.weight.requires_grad = False # FREEZE!
        self.pretrained.bias.requires_grad = False
        
        # 2. LoRA Adapters (Trainable)
        # B: Initialize to zeros
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        # A: Initialize with random gaussian
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        
        self.scaling = alpha / rank
        
        # Reset parameters
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # y = Wx + (B*A)*x * scale
        pretrained_out = self.pretrained(x)
        
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        
        return pretrained_out + lora_out * self.scaling

if __name__ == "__main__":
    # Test
    layer = LoRALinear(128, 64, rank=8)
    x = torch.randn(1, 128)
    y = layer(x)
    
    # Check gradients
    print("Pretrained grad:", layer.pretrained.weight.grad) # Should be None
    print("LoRA_A requires grad:", layer.lora_A.requires_grad) # True
    print(f"Output shape: {y.shape}")
