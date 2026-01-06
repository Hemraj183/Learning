import torch
import torch.nn as nn
from course_materials.week9_lora.project import LoRALinear
from course_materials.week10_moe.project import MoE

# ==========================================
# Final Capstone: The Efficient Assistant
# ==========================================
class UltimateAssistant(nn.Module):
    def __init__(self):
        super().__init__()
        print("Initializing Ultimate Assistant...")
        
        # 1. Efficient Core: MoE Layer
        # Replacing dense feed-forward with Sparse MoE
        self.brain = MoE(n_embed=128, num_experts=4)
        
        # 2. Skill Adapters: LoRA Layers
        # We can switch these 'skills' on the fly without reloading the full model
        self.skill_coding = LoRALinear(128, 128, rank=4)
        self.skill_creative = LoRALinear(128, 128, rank=4)
        
    def forward(self, x, skill="general"):
        # Process through core brain
        base_features = self.brain(x)
        
        # Apply specific skill adapter
        if skill == "coding":
            return self.skill_coding(base_features)
        elif skill == "creative":
            return self.skill_creative(base_features)
        else:
            return base_features

if __name__ == "__main__":
    assistant = UltimateAssistant()
    
    dummy_input = torch.randn(1, 10, 128)
    
    print("\n1. Asking coding question...")
    out_code = assistant(dummy_input, skill="coding")
    print(f"Output shape: {out_code.shape}")
    
    print("\n2. Asking creative writing...")
    out_creative = assistant(dummy_input, skill="creative")
    print("Done. System is modular and efficient!")
