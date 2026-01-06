import torch
import pytest
import os
from course_materials.week10_moe.project import MoE

def test_moe_output_shape():
    model = MoE(n_embed=32, num_experts=4)
    # Forward expects [batch, seq, n_embed]
    x = torch.randn(8, 5, 32)
    output = model(x)
    
    assert output.shape == (8, 5, 32)
    assert isinstance(output, torch.Tensor)