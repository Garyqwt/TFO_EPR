import torch.nn as nn
import torch

class FusionMLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FusionMLP, self).__init__()
        
        layers = []
        current_size = hidden_size
        
        # First layer
        layers.append(nn.Linear(input_size, current_size))
        layers.append(nn.ReLU())
        
        # Following layers, decreasing by a factor of 2 until 8
        while current_size > 8:
            next_size = max(8, current_size // 2)
            layers.append(nn.Linear(current_size, next_size))
            layers.append(nn.BatchNorm1d(next_size))
            layers.append(nn.ReLU())
            current_size = next_size
        
        # Final output layer
        layers.append(nn.Linear(current_size, 1))
        
        # Combine all layers
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)