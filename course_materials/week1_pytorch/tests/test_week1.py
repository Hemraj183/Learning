import torch
import torch.nn as nn
import pytest
import os

# Add parent directory to path to import project

from course_materials.week1_pytorch.project import ManualLinear, MLP, train_epoch

def test_manual_linear_initialization():
    layer = ManualLinear(10, 5)
    assert isinstance(layer.weight, nn.Parameter)
    assert isinstance(layer.bias, nn.Parameter)
    assert layer.weight.shape == (5, 10)  # PyTorch convention: (out, in)
    assert layer.bias.shape == (5,)

def test_manual_linear_forward():
    layer = ManualLinear(10, 5)
    x = torch.randn(32, 10)
    output = layer(x)
    assert output.shape == (32, 5)

def test_mlp_architecture():
    model = MLP()
    # Check layers exist
    assert hasattr(model, 'layer1')
    assert hasattr(model, 'layer2')
    assert hasattr(model, 'layer3')
    
    # Check dimensions
    assert model.layer1.weight.shape == (256, 784)
    assert model.layer2.weight.shape == (128, 256)
    assert model.layer3.weight.shape == (10, 128)

def test_mlp_forward():
    model = MLP()
    x = torch.randn(4, 1, 28, 28)  # Batch of images
    output = model(x)
    assert output.shape == (4, 10)

def test_training_step():
    # Mock data
    model = MLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cpu')
    
    # Create fake dataloader
    data = torch.randn(10, 1, 28, 28)
    target = torch.randint(0, 10, (10,))
    fake_loader = [(data, target)]
    
    loss, acc = train_epoch(model, fake_loader, criterion, optimizer, device)
    
    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert 0 <= acc <= 100