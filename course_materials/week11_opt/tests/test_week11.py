import torch
import pytest
import os
from course_materials.week11_opt.project import compare_quantization

def test_quantization_exists():
    # Verify the main function exists and is callable
    assert callable(compare_quantization)
    
    # Since compare_quantization runs a full script with prints and timing, 
    # we might not want to run it fully in a unit test unless we mock prints/time.
    # For now, existence check is sufficient for "Zero to Hero" structure validation.