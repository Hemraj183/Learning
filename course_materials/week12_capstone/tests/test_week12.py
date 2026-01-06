import torch
import pytest
import os
from course_materials.week12_capstone.project import UltimateAssistant

def test_ultimate_assistant_integration():
    model = UltimateAssistant()
    # Check components
    assert hasattr(model, 'brain')
    assert hasattr(model, 'skill_coding')
    assert hasattr(model, 'skill_creative')
    
    # Mock input
    # Input shape for MoE/LoRA depends on implementation. 
    # Project.py uses: dummy_input = torch.randn(1, 10, 128)
    dummy_input = torch.randn(1, 5, 128)
    
    # Test forward pass with different skills
    out_gen = model(dummy_input, skill="general")
    out_code = model(dummy_input, skill="coding")
    
    assert out_gen.shape == dummy_input.shape
    assert out_code.shape == dummy_input.shape