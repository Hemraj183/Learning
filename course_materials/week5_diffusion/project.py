import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# ==========================================
# 1. Diffusion Schedulers
# ==========================================
class DiffusionProcess:
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.timesteps = timesteps
        
        # Linear Beta Schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        
        # Alpha calculations
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Sqrt values for the forward process q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (Forward Process)
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise

# ==========================================
# 2. Visualization
# ==========================================
def visualize_diffusion():
    # Mock image (black square on white background)
    x_0 = torch.ones(1, 1, 32, 32)
    x_0[:, :, 8:24, 8:24] = 0 # Black box in middle
    x_0 = (x_0 * 2) - 1 # Normalize to [-1, 1]
    
    dp = DiffusionProcess(timesteps=300)
    
    # Visualize steps
    steps = [0, 50, 150, 299]
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    
    for i, step in enumerate(steps):
        t = torch.tensor([step])
        x_t = dp.q_sample(x_0, t)
        
        # Convert back to image
        img = x_t[0, 0].detach().numpy()
        axs[i].imshow(img, cmap='gray')
        axs[i].set_title(f"Step {step}")
        axs[i].axis('off')
        
    plt.show()
    print("Diffusion Visualization Complete.")

if __name__ == "__main__":
    import torch.nn.functional as F
    visualize_diffusion()
