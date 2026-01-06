import torch
import pytest
import os
from course_materials.week9_lora.project import LoRALinear

def test_lora_freeze():
    # Pass in_features, out_features integers
    lora = LoRALinear(10, 10, rank=4, alpha=8)
    
    # Check if original weights are frozen
    assert not lora.pretrained.weight.requires_grad
    
    # Check if A and B are trainable
    assert lora.lora_A.requires_grad
    assert lora.lora_B.requires_grad

def test_lora_forward():
    lora = LoRALinear(10, 10, rank=4, alpha=8)
    x = torch.randn(5, 10)
    y = lora(x)
    assert y.shape == (5, 10)
