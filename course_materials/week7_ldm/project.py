import torch
import torch.nn as nn

# ==========================================
# 0. Components Mockup (VAE & CLIP)
# ==========================================
class VAE_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Compresses 3x64x64 image to 4x8x8 latent
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 4, 3, 4, 1) # Crude downsampling
        )
    def forward(self, x): return self.net(x)

class VAE_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Expands 4x8x8 latent back to 3x64x64
        self.net = nn.Sequential(
            nn.ConvTranspose2d(4, 64, 3, 4, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, 2, 1, output_padding=1)
        )
    def forward(self, x): return self.net(x)

class CLIP_TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1000, 77) # Mock vocab
    def forward(self, x): return self.embedding(x)

# ==========================================
# 1. Latent Diffusion Model Class
# ==========================================
class LatentDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.vae_enc = VAE_Encoder()
        self.vae_dec = VAE_Decoder()
        self.text_enc = CLIP_TextEncoder()
        
        # The U-Net now operates on 4x8x8 Latents!
        # This makes it much faster than Pixel Diffusion
        self.unet = nn.Conv2d(4, 4, 3, 1, 1) # Mock U-Net for simplicity

    def forward(self, image, text_ids):
        # 1. Encode Image to Latent Space
        latents = self.vae_enc(image) # z = E(x)
        
        # 2. Encode Text
        context = self.text_enc(text_ids)
        
        # 3. Add Noise to Latents (Forward Process)
        noise = torch.randn_like(latents)
        noisy_latents = latents + noise # Simplified
        
        # 4. Predict Noise (Reverse Process)
        # In real LDM, cross-attention injects 'context' into UNet
        pred_noise = self.unet(noisy_latents)
        
        return pred_noise, noise

# ==========================================
# 2. Demo
# ==========================================
if __name__ == "__main__":
    model = LatentDiffusion()
    
    img = torch.randn(1, 3, 64, 64)
    text = torch.randint(0, 1000, (1, 5))
    
    pred_noise, real_noise = model(img, text)
    
    print(f"Image Shape: {img.shape}")
    print(f"Latent Shape: {model.vae_enc(img).shape} (Smaller!)")
    print("Forward pass successful.")
