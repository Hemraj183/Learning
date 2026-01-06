import torch
import pytest
import os
from course_materials.week4_router.project import RouterNetwork

def test_router_initialization():
    model = RouterNetwork(num_experts=3)
    assert model.classifier is not None
    assert hasattr(model, 'bert')

def test_router_routing():
    model = RouterNetwork(num_experts=3)
    # Use route() method which handles tokenization
    text = "Hello world"
    expert_id = model.route(text)
    assert isinstance(expert_id, int)
    assert 0 <= expert_id < 3
