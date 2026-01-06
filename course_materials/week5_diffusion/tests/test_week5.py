import torch
import pytest
import os
from course_materials.week5_diffusion.project import DiffusionProcess

def test_diffusion_schedule():
    diff = DiffusionProcess(timesteps=100)
    assert len(diff.betas) == 100
    assert len(diff.alphas) == 100
    
def test_forward_diffusion_shape():
    diff = DiffusionProcess(timesteps=100)
    x0 = torch.randn(1, 1, 32, 32)
    t = torch.tensor([10])
    
    # Method is q_sample
    xt = diff.q_sample(x0, t)
    
    # Returns just x_t (noise is implicit or passed explicitly)
    assert xt.shape == x0.shape
