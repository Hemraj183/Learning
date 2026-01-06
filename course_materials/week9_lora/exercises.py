import torch
from project import LoRALinear

def exercise_merge():
    # TODO: Implement a 'merge' method to collapse LoRA weights back into the main Linear layer
    # W_new = W_old + (B @ A) * scale
    # This makes inference faster (no extra matrix mul)
    print("Exercise: Implement LoRA Weight Merging.")

if __name__ == "__main__":
    exercise_merge()
