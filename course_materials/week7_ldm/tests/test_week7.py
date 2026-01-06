import torch
import pytest
import os
from course_materials.week7_ldm.project import LatentDiffusion

def test_ldm_components():
    model = LatentDiffusion()
    assert hasattr(model, 'vae_enc')
    assert hasattr(model, 'unet')
    
def test_ldm_forward():
    model = LatentDiffusion()
    # Mock text and image
    # Text encoder expects tensor indices [batch, seq]
    text_ids = torch.randint(0, 1000, (1, 10))
    image = torch.randn(1, 3, 64, 64)
    
    pred_noise, noise = model(image, text_ids)
    
    # Output shape should be latent shape (4x8x8)
    # VAE encodes 64x64 -> 8x8 (stride 8 total? Let's check logic roughly)
    # VAE: 3x3 s2 -> 3x3 s4 -> 8x downsampling? 64/2/4 = 8. Correct.
    assert pred_noise.shape == (1, 4, 8, 8)
