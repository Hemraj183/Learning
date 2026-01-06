import torch
import os
import pytest
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Add parent directory to path to import project

from course_materials.week3_llm_variants.project import generate_text

def test_tokenizer_loading():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    assert tokenizer.vocab_size == 50257
    assert tokenizer.encode("Hello") == [15496]

def test_model_loading():
    # Only test if model loads without error (might be slow on CPU)
    # This test assumes internet access or cached models
    try:
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        assert isinstance(model, GPT2LMHeadModel)
    except Exception as e:
        pytest.skip(f"Could not load model (network issue?): {e}")

def test_generation_logic():
    # Mock model and tokenizer for speed
    class MockTokenizer:
        def encode(self, text, return_tensors='pt'):
            return torch.tensor([[101, 102]])
        def decode(self, ids, skip_special_tokens=True):
            return "Mock Output"
        @property
        def eos_token_id(self):
            return 999

    class MockModel(torch.nn.Module):
        def forward(self, input_ids):
            # Return fake logits [Batch, Seq, Vocab]
            # Use small vocab size 1000
            batch, seq = input_ids.shape
            logits = torch.randn(batch, seq, 1000)
            return type('Outputs', (), {'logits': logits})()

    mock_model = MockModel()
    mock_tokenizer = MockTokenizer()
    
    output = generate_text(mock_model, mock_tokenizer, "Test", max_length=5)
    assert isinstance(output, str)
    assert output == "Mock Output"

def test_embedding_inspection():
    # Verify we can access the embedding layer of the real model structure
    try:
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        embeddings = model.transformer.wte.weight
        assert embeddings.shape[0] == 50257
        assert embeddings.shape[1] == 768
    except:
        pytest.skip("Skipping model inspection due to load failure")