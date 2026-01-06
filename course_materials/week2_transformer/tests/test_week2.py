import torch
import torch.nn as nn
import time
import os
import math

# Add parent directory to path to import project

from course_materials.week2_transformer.project import SelfAttention, MultiHeadAttention, TransformerBlock

def test_self_attention_shape():
    B, S, D = 2, 10, 64
    x = torch.randn(B, S, D)
    sa = SelfAttention(D, 32)
    out, scores = sa(x)
    
    assert out.shape == (B, S, 32)
    assert scores.shape == (B, S, S)
    
    # Check if attention scores sum to 1
    assert torch.allclose(scores.sum(dim=-1), torch.ones(B, S), atol=1e-5)

def test_multi_head_attention_shape():
    B, S, D = 4, 16, 128
    x = torch.randn(B, S, D)
    mha = MultiHeadAttention(D, 4)
    out = mha(x)
    
    assert out.shape == (B, S, D)

def test_transformer_block_forward():
    B, S, D = 2, 20, 256
    x = torch.randn(B, S, D)
    block = TransformerBlock(D, 8)
    out = block(x)
    
    assert out.shape == (B, S, D)

def test_scaled_dot_product_math():
    # Specific math check
    # if q=k, and v is identity, output should be proportional
    B, S, D = 1, 1, 16
    q = torch.ones(B, S, D)
    k = torch.ones(B, S, D)
    v = torch.ones(B, S, D) * 2
    
    # Custom simple SDPA for check
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
    attn = torch.softmax(scores, dim=-1)
    expected = torch.matmul(attn, v)
    
    # This is more of a sanity check on the concept
    assert expected.shape == (B, S, D)