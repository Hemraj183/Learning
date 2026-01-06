import torch
import torch.nn as nn
import time

def compare_quantization():
    print("Comparing FP32 vs INT8...")
    
    # 1. Create a large model
    model_fp32 = nn.Sequential(
        nn.Linear(1024, 4096),
        nn.ReLU(),
        nn.Linear(4096, 1024)
    )
    
    input_data = torch.randn(100, 1024)
    
    # 2. Measure FP32 Size & Speed
    start = time.time()
    for _ in range(100):
        _ = model_fp32(input_data)
    fp32_time = time.time() - start
    
    fp32_size = sum(p.numel() * 4 for p in model_fp32.parameters()) / 1024 / 1024
    
    print(f"FP32 Size: {fp32_size:.2f} MB")
    print(f"FP32 Time: {fp32_time:.4f} s")
    
    # 3. Dynamic Quantization (INT8)
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32, 
        {nn.Linear}, 
        dtype=torch.qint8
    )
    
    start = time.time()
    for _ in range(100):
        _ = model_int8(input_data)
    int8_time = time.time() - start
    
    # Size estimate (rough)
    # Weights are 1/4th size
    int8_size = fp32_size / 4 
    
    print(f"INT8 Size (Est): {int8_size:.2f} MB")
    print(f"INT8 Time: {int8_time:.4f} s")
    print(f"Speedup: {fp32_time / int8_time:.2f}x")

if __name__ == "__main__":
    compare_quantization()
