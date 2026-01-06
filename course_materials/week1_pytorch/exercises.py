import torch
import torch.nn as nn

# ==========================================
# Exercise 1: Gradient Debugging
# ==========================================
def gradient_debugging_demo():
    print("--- Exercise 1: Gradient Debugging ---")
    x = torch.tensor(2.0, requires_grad=True)
    
    # Pass 1
    y = x ** 2
    y.backward()
    print(f"Gradient after pass 1: {x.grad}")
    
    # Fix this code!
    # Problem: x.grad is accumulating
    # TODO: Add the fix here
    
    # Pass 2
    z = x ** 3
    z.backward()
    print(f"Gradient after pass 2: {x.grad}")
    print("(Should be 12.0 if fixed, but is likely 16.0)")

# ==========================================
# Exercise 2: Add Dropout
# ==========================================
class MLPWithDropout(nn.Module):
    """
    TODO: Modify this class to include Dropout layers
    Args:
        dropout_p (float): Probability of zeroing an element
    """
    def __init__(self, dropout_p=0.2):
        super().__init__()
        # TODO: Define layers
        pass
    
    def forward(self, x):
        # TODO: Implement forward pass with dropout
        pass

# ==========================================
# Exercise 3: Batch Normalization
# ==========================================
class MLPWithBatchNorm(nn.Module):
    """
    TODO: Add Batch Normalization layers
    Hint: nn.BatchNorm1d(num_features)
    """
    def __init__(self):
        super().__init__()
        # TODO: Define layers
        pass
    
    def forward(self, x):
        # TODO: Implement forward pass with BatchNorm
        pass

# ==========================================
# Challenge: He Initialization
# ==========================================
class HeLinear(nn.Module):
    """
    TODO: Implement He Initialization instead of Xavier
    He Init: std = sqrt(2 / fan_in)
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features)) # TODO: Scale this properly
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        return x @ self.weight.T + self.bias

if __name__ == '__main__':
    gradient_debugging_demo()
