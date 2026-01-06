import torch
import torch.nn as nn
# Import Previous Weeks
from course_materials.week4_router.project import RouterNetwork
from course_materials.week6_unet.project import UNet

# ==========================================
# Capstone 2: The "Noisy" Router
# ==========================================
class NoisyRouter(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. The Decision Maker
        self.router = RouterNetwork(num_experts=2) # 0: Generate Image, 1: Describe Image
        
        # 2. Expert A: The Artist (Small Diffusion Model)
        self.diffusion = UNet(in_channels=3, out_channels=3)
        
        # 3. Expert B: The Critic (Captioner - Mock)
        self.captioner = nn.Linear(64, 10) 

    def forward(self, text_input, image_input=None):
        # Step 1: Decide what to do
        # In a real app, we'd use the tokenizer from Week 4
        # Here we simulate the router output for demonstration
        print(f"Routing query: '{text_input}'...")
        
        if "draw" in text_input or "generate" in text_input:
            route = 0 # Artist
        else:
            route = 1 # Critic
            
        if route == 0:
            print(">> Selected Expert: Artist (Diffusion)")
            # Generate random noise and denoise it (one step mock)
            noise = torch.randn(1, 3, 64, 64)
            output = self.diffusion(noise)
            return "Generated Image Tensor", output
        else:
            print(">> Selected Expert: Critic (Captioning)")
            return "This is a masterpiece.", None

if __name__ == "__main__":
    system = NoisyRouter()
    
    # Test Flow 1
    resp, img = system("draw a cat")
    print(f"Result: {resp}\n")
    
    # Test Flow 2
    resp, _ = system("explain this image")
    print(f"Result: {resp}\n")
