import torch
import pytest
import os
from course_materials.week8_capstone.project import NoisyRouter

def test_noisy_router_routing():
    model = NoisyRouter()
    
    # Case 1: Chat (Critic)
    out_msg, out_tensor = model("Tell me a joke")
    assert isinstance(out_msg, str)
    assert out_tensor is None
    
    # Case 2: Draw (Artist)
    # Note: 'draw' triggers the artist
    out_msg_2, out_tensor_2 = model("draw a cat")
    assert isinstance(out_msg_2, str)
    assert out_tensor_2 is not None
    assert out_tensor_2.shape == (1, 3, 64, 64)
