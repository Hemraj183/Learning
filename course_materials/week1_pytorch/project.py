import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. Manual Linear Layer
# ==========================================
class ManualLinear(nn.Module):
    """Linear layer implemented with nn.Parameter"""
    def __init__(self, in_features, out_features):
        super().__init__()
        # Xavier initialization
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * (2.0 / in_features) ** 0.5
        )
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        # x: (batch_size, in_features)
        # weight: (out_features, in_features)
        # output: (batch_size, out_features)
        return x @ self.weight.T + self.bias

# ==========================================
# 2. MLP Architecture
# ==========================================
class MLP(nn.Module):
    """Multi-Layer Perceptron for MNIST"""
    def __init__(self):
        super().__init__()
        self.layer1 = ManualLinear(784, 256)
        self.layer2 = ManualLinear(256, 128)
        self.layer3 = ManualLinear(128, 10)
    
    def forward(self, x):
        # Flatten input: (batch, 1, 28, 28) -> (batch, 784)
        x = x.view(x.size(0), -1)
        
        # Layer 1
        x = self.layer1(x)
        x = torch.relu(x)
        
        # Layer 2
        x = self.layer2(x)
        x = torch.relu(x)
        
        # Layer 3 (output)
        x = self.layer3(x)
        
        return x  # Raw logits (CrossEntropyLoss applies softmax)

# ==========================================
# 3. Training & Evaluation Functions
# ==========================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy

# ==========================================
# 4. Main Execution
# ==========================================
def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data Loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Check if data exists or download
    try:
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Model Init
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Training Loop
    num_epochs = 3  # Reduced for quick testing, increase for full training
    
    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}')
        print('-' * 50)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    # Save
    torch.save(model.state_dict(), 'mnist_mlp.pth')
    print('\nModel saved to mnist_mlp.pth')

if __name__ == '__main__':
    main()
