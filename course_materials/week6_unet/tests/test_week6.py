import torch
import pytest
import os


from course_materials.week6_unet.project import UNet

def test_unet_shape():
    model = UNet(in_channels=3, out_channels=3)
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert out.shape == x.shape